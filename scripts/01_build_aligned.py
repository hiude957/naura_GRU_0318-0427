#!/usr/bin/env python
"""Build fixed 110 ms aligned tables."""

import argparse

from naura_gru.data.grid_align import build_aligned_range
from naura_gru.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/align_110ms.yaml")
    parser.add_argument("--date-start")
    parser.add_argument("--date-end")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.date_start:
        config["date_start"] = args.date_start
    if args.date_end:
        config["date_end"] = args.date_end
    build_aligned_range(config)


if __name__ == "__main__":
    main()
