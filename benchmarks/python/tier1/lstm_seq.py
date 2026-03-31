"""LSTM sequence benchmark: multi-layer LSTM + output projection.

Directly comparable to gru_seq — same dimensions, different cell type.
"""

import torch
import torch.nn as nn


SEQ_LEN = 50
INPUT_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2


class LstmSeqModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True)
        self.output = nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def forward(self, x):
        # x: [B, seq_len, input_dim]
        output, _ = self.lstm(x)
        # Take last timestep: [B, hidden_dim]
        return self.output(output[:, -1, :])


def run(device, batches_per_epoch=50, batch_size=128, **kwargs):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from harness import run_benchmark

    # Disable cuDNN benchmark for LSTM: cell-level unrolling (in flodl)
    # causes variance; disable for both sides for fair comparison.
    prev_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False

    model = LstmSeqModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batches = [
        (torch.randn(batch_size, SEQ_LEN, INPUT_DIM, device=device),
         torch.randn(batch_size, INPUT_DIM, device=device))
        for _ in range(batches_per_epoch)
    ]

    result = run_benchmark("lstm_seq", model, batches, device, optimizer, **kwargs)
    torch.backends.cudnn.benchmark = prev_benchmark  # restore for subsequent benchmarks
    return result
