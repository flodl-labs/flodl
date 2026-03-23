"""GRU sequence benchmark: GRUCell unrolled over timesteps + output projection."""

import torch
import torch.nn as nn


SEQ_LEN = 50
INPUT_DIM = 256
HIDDEN_DIM = 512


class GruSeqModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(INPUT_DIM, HIDDEN_DIM)
        self.output = nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def forward(self, x):
        # x: [B, seq_len, input_dim]
        batch = x.size(0)
        h = torch.zeros(batch, HIDDEN_DIM, device=x.device)
        for t in range(SEQ_LEN):
            h = self.gru(x[:, t, :], h)
        return self.output(h)


def run(device, batches_per_epoch=50, batch_size=128, **kwargs):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from harness import run_benchmark

    # Disable cuDNN benchmark for GRU: cell-level unrolling triggers
    # nondeterministic autotuning that causes massive run-to-run variance.
    prev_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False

    model = GruSeqModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batches = [
        (torch.randn(batch_size, SEQ_LEN, INPUT_DIM, device=device),
         torch.randn(batch_size, INPUT_DIM, device=device))
        for _ in range(batches_per_epoch)
    ]

    result = run_benchmark("gru_seq", model, batches, device, optimizer, **kwargs)
    torch.backends.cudnn.benchmark = prev_benchmark  # restore for subsequent benchmarks
    return result
