#!/usr/bin/env python
"""Build cache arrays from aligned fixed-grid tables."""

import argparse

from naura_gru.data.cache_builder import build_cache
from naura_gru.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cache_dataset.yaml")
    args = parser.parse_args()
    build_cache(load_config(args.config))


if __name__ == "__main__":
    main()

