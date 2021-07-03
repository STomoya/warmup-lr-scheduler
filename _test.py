
import torch
import torch.nn as nn
import torch.optim as optim
from scheduler import *

if __name__=='__main__':
    epochs = 100
    per_batch = 100
    iterations = epochs * per_batch

    param = [nn.Parameter(torch.randn(10))]
    opt_linear_lin = optim.Adam(param)
    opt_linear_cos = optim.Adam(param)
    opt_linear_exp = optim.Adam(param)
    opt_linear_ms  = optim.Adam(param)
    opt_const_lin = optim.Adam(param)
    opt_const_cos = optim.Adam(param)
    opt_const_exp = optim.Adam(param)
    opt_const_ms  = optim.Adam(param)

    base = LinearMultiplier(start=1., end=0.0001)
    multiplier = with_warmup(base, 0.001, 0.1)
    sch_linear_lin = MultiplyLR(
        opt_linear_lin, multiplier, iterations)
    multiplier = with_warmup(base, 0.001, 0.1, 'constant')
    sch_const_lin = MultiplyLR(
        opt_const_lin, multiplier, iterations)

    base = CosineMultiplier(start=1, min_value=0.0001)
    multiplier = with_warmup(base, 0.001, 0.1)
    sch_linear_cos = MultiplyLR(
        opt_linear_cos, multiplier, iterations)
    multiplier = with_warmup(base, 0.001, 0.1, 'constant')
    sch_const_cos = MultiplyLR(
        opt_const_cos, multiplier, iterations)

    base = ExponentialMultiplier(start=1., decay=0.02)
    multiplier = with_warmup(base, 0.001, 0.1)
    sch_linear_exp = MultiplyLR(
        opt_linear_exp, multiplier, iterations)
    multiplier = with_warmup(base, 0.001, 0.1, 'constant')
    sch_const_exp = MultiplyLR(
        opt_const_exp, multiplier, iterations)

    base = MultiStepMultiplier(
        list(map(int, [iterations*0.8, iterations*0.9])),
        iterations, gamma=0.1)
    multiplier = with_warmup(base, 0.001, 0.1)
    sch_linear_ms = MultiplyLR(
        opt_linear_ms, multiplier, iterations)
    multiplier = with_warmup(base, 0.001, 0.1, 'constant')
    sch_const_ms = MultiplyLR(
        opt_const_ms, multiplier, iterations)

    lrs = {
        'const_lin': [], 'linear_lin': [],
        'const_cos': [], 'linear_cos': [],
        'const_exp': [], 'linear_exp': [],
        'const_ms': [],  'linear_ms': []
    }
    titles = {
        'const_lin': 'linear learning rate schedule + constant warmup',
        'linear_lin': 'linear learning rate schedule + linear warmup',
        'const_cos': 'cosine learning rate schedule + constant warmup',
        'linear_cos': 'cosine learning rate schedule + linear warmup',
        'const_exp': 'exponential learning rate schedule + constant warmup',
        'linear_exp': 'exponential learning rate schedule + linear warmup',
        'const_ms': 'multi-step learning rate schedule + constant warmup',
        'linear_ms': 'multi-step learning rate schedule + linear warmup'
    }
    for epoch in range(epochs):
        for batch in range(per_batch):
            sch_const_lin.step()
            lrs['const_lin'].append(sch_const_lin.get_last_lr())
            sch_const_cos.step()
            lrs['const_cos'].append(sch_const_cos.get_last_lr())
            sch_const_exp.step()
            lrs['const_exp'].append(sch_const_exp.get_last_lr())
            sch_const_ms.step()
            lrs['const_ms'].append(sch_const_ms.get_last_lr())
            sch_linear_lin.step()
            lrs['linear_lin'].append(sch_linear_lin.get_last_lr())
            sch_linear_cos.step()
            lrs['linear_cos'].append(sch_linear_cos.get_last_lr())
            sch_linear_exp.step()
            lrs['linear_exp'].append(sch_linear_exp.get_last_lr())
            sch_linear_ms.step()
            lrs['linear_ms'].append(sch_linear_ms.get_last_lr())

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 16))
    x = range(iterations)

    for i, key in enumerate(lrs.keys(), 1):
        ax = fig.add_subplot(4, 2, i)
        ax.plot(x, lrs[key])
        plt.xlabel('iterations')
        plt.ylabel('learning rate')
        plt.title(titles[key])

    plt.tight_layout()
    plt.savefig('lr')
