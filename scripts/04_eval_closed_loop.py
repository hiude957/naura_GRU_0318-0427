#!/usr/bin/env python
"""Evaluate closed-loop sensor prediction."""

import argparse

from naura_gru.evaluation.closed_loop import evaluate_closed_loop
from naura_gru.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_closed_loop.yaml")
    args = parser.parse_args()
    evaluate_closed_loop(load_config(args.config))


if __name__ == "__main__":
    main()

