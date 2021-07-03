'''
Learning rate scheduler.
Code is taken from fvcore: https://github.com/facebookresearch/fvcore
and modified by STomoya: https://github.com/STomoya
'''

import math
from typing import List

import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

__all__ = [
    'ConstantMultiplier',
    'LinearMultiplier',
    'CosineMultiplier',
    'ExponentialMultiplier',
    'MultiStepMultiplier',
    'Compose',
    'MultiplyLR',
    'with_warmup'
]

class Multiplier:
    WHERE_EPSILON = 1e-6
    def __call__(self, where: float) -> float:
        raise NotImplementedError()

class ConstantMultiplier(Multiplier):
    def __init__(self,
        value: float
    ) -> None:
        self.value = value
    def __call__(self, where: float) -> float:
        return self.value

class LinearMultiplier(Multiplier):
    def __init__(self,
        start: float,
        end: float
    ) -> None:
        self.start = start
        self.end = end

    def __call__(self, where: float) -> float:
        return self.end * where + self.start * (1 - where)

class CosineMultiplier(Multiplier):
    def __init__(self,
        start: float,
        min_value: float,
        T_max: float=1.
    ) -> None:
        self.start = start
        self.min_value = min_value
        self.T_max = T_max

    def __call__(self, where: float) -> float:
        return self.min_value \
            + 0.5 * (self.start - self.min_value) \
            * (1 + math.cos(math.pi * where / self.T_max))

class ExponentialMultiplier(Multiplier):
    def __init__(self,
        start: float,
        decay: float
    ) -> None:
        self.start = start
        self.decay = decay
    def __call__(self, where: float) -> float:
        return self.start * (self.decay ** where)

class MultiStepMultiplier(Multiplier):
    def __init__(self,
        milestones: List[int],
        max_iters: int,
        gamma: float,
        initial_scale: float=1.
    ) -> None:
        self.milestones = milestones
        self.max_iters = max_iters
        self.gamma = gamma
        self.curret = initial_scale

    def __call__(self, where: float) -> float:
        epoch_num = int((where + self.WHERE_EPSILON) * self.max_iters)
        if epoch_num in self.milestones:
            self.curret *= self.gamma
        return self.curret

class Compose(Multiplier):
    def __init__(self,
        multipliers: List[Multiplier],
        lengths: List[int],
        scaling: List[str]
    ) -> None:
        assert len(multipliers) == len(lengths)
        assert 0 <= (sum(lengths) - 1.) < 1e-3
        assert all([s in ['scaled', 'fixed'] for s in scaling])

        self.multipliers = multipliers
        self.lengths = lengths
        self.scaling = scaling

    def __call__(self, where: float) -> float:
        running_total = 0.
        for i, length in enumerate(self.lengths):
            running_total += length
            if where + self.WHERE_EPSILON <= running_total:
                break

        multiplier = self.multipliers[i]

        if self.scaling[i] == 'scaled':
            start = running_total - self.lengths[i]
            where = (where - start) / self.lengths[i]

        return multiplier(where)

def with_warmup(
    multiplier: Multiplier,
    warmup_factor: float,
    warmup_length: float,
    warmup_method: str='linear'
) -> Compose:
    assert warmup_method in ['linear', 'constant']
    end = multiplier(warmup_length)
    start = warmup_factor * multiplier(0.)
    if warmup_method == 'linear':
        warmup = LinearMultiplier(start, end)
    elif warmup_method == 'constant':
        warmup = ConstantMultiplier(start)

    return Compose(
        [warmup, multiplier],
        [warmup_length, (1 - warmup_length)],
        ['scaled', 'fixed']
    )

class MultiplyLR(_LRScheduler):
    def __init__(self,
        optimizer: optim.Optimizer,
        multiplier: Multiplier,
        max_iter: int,
        last_iter: int=-1
    ) -> None:
        self._multiplier = multiplier
        self._max_iter = max_iter
        super().__init__(optimizer, last_epoch=last_iter)

    def state_dict(self) -> list:
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}

    def get_lr(self) -> float:
        multiplier = self._multiplier(self.last_epoch / self._max_iter)
        return [base_lr * multiplier for base_lr in self.base_lrs]
