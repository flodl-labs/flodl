"""Convolutional autoencoder benchmark: Conv2d encoder + ConvTranspose2d decoder.

Tests transposed convolution throughput for image reconstruction.
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: stride-2 Conv2d, 64x64 → 4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),                                  # 64x64 → 32x32
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),                                  # 32x32 → 16x16
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),                                  # 16x16 → 8x8
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),                                  # 8x8 → 4x4
        )
        # Decoder: stride-2 ConvTranspose2d, 4x4 → 64x64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),                                  # 4x4 → 8x8
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),                                  # 8x8 → 16x16
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),                                  # 16x16 → 32x32
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh(),                                  # 32x32 → 64x64
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def run(device, batches_per_epoch=50, batch_size=64, **kwargs):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from harness import run_benchmark

    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Autoencoder: target = input (reconstruction)
    batches = [
        (torch.randn(batch_size, 3, 64, 64, device=device),) * 2
        for _ in range(batches_per_epoch)
    ]

    return run_benchmark("conv_autoenc", model, batches, device, optimizer, **kwargs)
