"""Fixed-iteration feedback loop: encoder -> loop(refine, N) -> decoder.

Same architecture as feedback_loop but always runs exactly MAX_ITER iterations
with no halt check and no GPU->CPU sync per iteration. Isolates pure framework
overhead for looping from adaptive halt behavior.
"""

import torch
import torch.nn as nn


DIM = 512
MAX_ITER = 10


class RefineBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.GELU(),
            nn.LayerNorm(DIM),
        )

    def forward(self, x):
        return x + self.net(x)


class FeedbackFixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.GELU(),
            nn.LayerNorm(DIM),
        )
        self.refine = RefineBlock()
        self.decoder = nn.Linear(DIM, DIM)

    def forward(self, x):
        x = self.encoder(x)
        for _ in range(MAX_ITER):
            x = self.refine(x)
        return self.decoder(x)


def run(device, batches_per_epoch=50, batch_size=128, **kwargs):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from harness import run_benchmark

    model = FeedbackFixed().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batches = [
        (torch.randn(batch_size, DIM, device=device),
         torch.randn(batch_size, DIM, device=device))
        for _ in range(batches_per_epoch)
    ]

    return run_benchmark("feedback_fixed", model, batches, device, optimizer, **kwargs)
