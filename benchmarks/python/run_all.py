#!/usr/bin/env python3
"""Run all PyTorch benchmarks and output JSON results."""

import json
import sys

from harness import get_device, print_result, print_summary

from tier1 import mlp, convnet, gru_seq, transformer, lstm_seq, conv_autoencoder
from tier2 import residual_tower, gated_routing, iterative_refine, feedback_loop_fixed


BENCHMARKS = {
    # Tier 1
    "mlp": mlp,
    "convnet": convnet,
    "gru_seq": gru_seq,
    "transformer": transformer,
    "lstm_seq": lstm_seq,
    "conv_autoenc": conv_autoencoder,
    # Tier 2
    "residual_tower": residual_tower,
    "gated_routing": gated_routing,
    "iterative_refine": iterative_refine,
    "feedback_fixed": feedback_loop_fixed,
}

TIER1 = {"mlp", "convnet", "gru_seq", "transformer", "lstm_seq", "conv_autoenc"}


def main():
    args = sys.argv[1:]
    output_json = "--json" in args
    tier1_only = "--tier1" in args
    tier2_only = "--tier2" in args
    single = None
    if "--bench" in args:
        idx = args.index("--bench")
        if idx + 1 < len(args):
            single = args[idx + 1]

    device = get_device()
    run_all = not tier1_only and not tier2_only and single is None

    results = []
    for name, module in BENCHMARKS.items():
        if single is not None:
            if name != single:
                continue
        elif not run_all:
            if tier1_only and name not in TIER1:
                continue
            if tier2_only and name in TIER1:
                continue

        print(f"--- {name} ---", file=sys.stderr)
        try:
            result = module.run(device)
            if not output_json:
                print_result(result)
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            print(file=sys.stderr)

    if not output_json and len(results) > 1:
        print_summary(results)

    if output_json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
