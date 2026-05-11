#!/usr/bin/env python
"""Train the GRU model."""

import argparse

from naura_gru.training.trainer import train
from naura_gru.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_gru_a100.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()

