
# warmup-lr-scheduler

Learning rate scheduler with warmup in PyTorch.

The codes are mostly taken from [fvcore](https://github.com/facebookresearch/fvcore)'s learning rate scheduler.

## Usage

`scheduler.step()` should be called every training step (not every epoch).

```python
import torch
from scheduler import *

epochs: int = 100
# these should be implemented by yourself
dataset: torch.utils.DataLoader = get_dataset(...)
model: torch.nn.Module = get_net(...)
optimizer: torch.optim.Optimizer = get_optim(...)
criterion: torch.nn.Module = get_criterion(...)

'''Scheduler'''
# linearly decay from 'start' to 'end'
multiplier = LinearMultiplier(
    start=1.,
    end=0.0001)
# add warmup
warmup_multiplier = with_warmup(
    multiplier,
    warmup_factor=0.0001,   # warmup start
    warmup_length=0.05,     # warmup length in [0, 1]
    warmup_method='linear') # 'linear' or 'constant'
# scheduler with warmup
scheduler = MultiplyLR(
    optimizer, warmup_multiplier, epochs*len(dataset))
'''---------'''

# training
for _ in range(epochs):
    for data, target in dataset:
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # should be called every training step
        scheduler.step()
```

### Multipliers

- ConstantMultiplier

```python
multiplier = ConstantMutiplier(
    value = 0.1
)
```

- LinearMultiplier

Can be either `start < end` for warmup or `end < start` for linear decay.

```python
multiplier = LinearMultiplier(
    start = 1.,
    end   = 0.0001
)
```

- ExponentialMultiplier

```python
multiplier = ExponentialMultiplier(
    start = 1.,
    decay = 0.02
)
```

- CosineMultiplier

```python
multiplier = ConsineMultiplier(
    start     = 1.,
    min_value = 0.0001, # minimum value
    T_max     = 1.      # same as T_max in torch.optim.lr_scheduler.CosineAnnealingLR
                        # but between [0, 1]
)
```

- MultiStepMultiplier

```python
multiplier = MultiStepMultiplier(
    milestones    = [800, 900], # milestones for updating learning rate in int
    max_iters     = 1000,       # total iterations for training
    gamma         = 0.1,        # gamma
    initial_scale = 1.          # initial scale
)
```

- Compose

Stick multiple `Multiplier` to one.

```python
multiplier = Compose(
    multipliers = [
        LinearMultiplier(...),
        CosineMultiplier(...)],        # a list of multipliers
    lengths     = [0.05, 0.95],        # length of each multiplier
                                       # should satisfy sum(lengths) == 1.
    scaling     = ['scaled', 'scaled'] # scaling method for each multiplier.
                                       # either 'scaled' or 'fixed'
)
```

- with_warmup

Add warmup to `Multiplier` using `Compose`.

```python
multiplier = with_warmup(
    multiplier,
    warmup_factor = 0.0001,  # start value for warmup
    warmup_length = 0.05,    # warmup length in [0, 1]
    warmup_method = 'linear' # either 'linear' or 'constant'
)
```

### Scheduler

Scheduler is made based on iterations, not epochs. Therefore `.step()` should be called every training step.

- MultiplyLR

```python
scheduler = MultiplyLR(
    optimizer,       # optimizer to update learning rate
    multiplier,      # multiplier
    max_iter = 1000, # total iterations to train
    last_iter = -1
)
```

### Sample

Some visualizations of learning rate using the scheduler.
Produced by `_test.py`.

![](lr.png)

## License

[MIT](LICENSE)

## Author

[STomoya](https://github.com/STomoya)
