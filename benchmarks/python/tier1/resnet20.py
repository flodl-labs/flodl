"""ResNet-20 benchmark: CIFAR-shaped input, SGD with momentum and weight decay.

The launch-heavy case: 65 small parameter tensors and a step of a few
milliseconds, so per-step framework overhead is a visible share of the time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class ResNet20(nn.Module):
    """ResNet-20 for 32x32 inputs (He et al. 2015)."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)
        blocks = []
        for in_ch, out_ch, stride in [(16, 16, 1), (16, 32, 2), (32, 64, 2)]:
            blocks.append(Block(in_ch, out_ch, stride))
            blocks += [Block(out_ch, out_ch, 1) for _ in range(2)]
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.blocks(F.relu(self.stem_bn(self.stem(x))))
        x = F.adaptive_avg_pool2d(x, 1)
        return self.fc(torch.flatten(x, 1))


def run(device, batches_per_epoch=100, batch_size=64, **kwargs):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from harness import run_benchmark

    model = ResNet20().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    batches = [
        (torch.randn(batch_size, 3, 32, 32, device=device),
         torch.randint(0, 10, (batch_size,), device=device))
        for _ in range(batches_per_epoch)
    ]

    return run_benchmark("resnet20", model, batches, device, optimizer,
                         loss_fn=F.cross_entropy, **kwargs)
