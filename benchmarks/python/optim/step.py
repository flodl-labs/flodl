"""Optimizer-step benchmarks: the step alone, no forward and no backward.

The tensors are ResNet-20's 65 parameters with a fixed gradient set once,
so a row prices what one `step()` costs for a model made of many small
tensors. Each optimizer runs with PyTorch's defaults, which is what a user
gets: the multi-tensor ("foreach") implementation on CUDA.
"""

import torch

from tier1.resnet20 import ResNet20

# Steps per measured "epoch". The comparison reports time per step.
STEPS = 200

# A small rate keeps the weights finite over tens of thousands of steps on
# a constant gradient; the step's cost does not depend on it.
LR = 1e-6

OPTIMIZERS = {
    "optim_sgd": lambda p: torch.optim.SGD(p, lr=LR, momentum=0.9, weight_decay=1e-4),
    "optim_adam": lambda p: torch.optim.Adam(p, lr=LR),
    "optim_adamw": lambda p: torch.optim.AdamW(p, lr=LR, weight_decay=1e-2),
    "optim_rmsprop": lambda p: torch.optim.RMSprop(p, lr=LR),
    "optim_adagrad": lambda p: torch.optim.Adagrad(p, lr=LR),
    "optim_nadam": lambda p: torch.optim.NAdam(p, lr=LR),
    "optim_radam": lambda p: torch.optim.RAdam(p, lr=LR),
}


def run(name, device, **kwargs):
    from harness import run_epochs

    params = list(ResNet20().to(device).parameters())
    # A fixed gradient, never zeroed: every step sees the same one.
    for p in params:
        p.grad = torch.randn_like(p)

    optimizer = OPTIMIZERS[name](params)

    def run_epoch():
        for _ in range(STEPS):
            optimizer.step()
        return 0.0

    return run_epochs(name, run_epoch, params, device,
                      batches_per_epoch=STEPS, batch_size=1, **kwargs)
