"""Transformer encoder benchmark: Embedding + N x (MHA + FFN + LayerNorm) + projection.

Tests attention throughput with cross-entropy loss on token predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB = 8192
D_MODEL = 512
D_FF = 2048
HEADS = 8
LAYERS = 4
SEQ_LEN = 128


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(D_MODEL, HEADS, batch_first=True)
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.ff1 = nn.Linear(D_MODEL, D_FF)
        self.ff2 = nn.Linear(D_FF, D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)

    def forward(self, x):
        attn, _ = self.mha(x, x, x)
        x = self.norm1(x + attn)
        ff = self.ff2(F.gelu(self.ff1(x)))
        return self.norm2(x + ff)


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, D_MODEL)
        self.pos_embed = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL))
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(LAYERS)])
        self.output = nn.Linear(D_MODEL, VOCAB)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def run(device, batches_per_epoch=50, batch_size=32, **kwargs):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from harness import run_benchmark

    model = TransformerEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    batches = [
        (torch.randint(0, VOCAB, (batch_size, SEQ_LEN), device=device),
         torch.randint(0, VOCAB, (batch_size, SEQ_LEN), device=device))
        for _ in range(batches_per_epoch)
    ]

    def loss_fn(pred, target):
        # pred: [B, seq, vocab], target: [B, seq]
        return F.cross_entropy(pred.reshape(-1, VOCAB), target.reshape(-1))

    return run_benchmark("transformer", model, batches, device, optimizer,
                         loss_fn=loss_fn, **kwargs)
