#!/usr/bin/env python3
import argparse
import gzip
import pickle
import sys
from pathlib import Path

from acn_embed.util import utils
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-pklgz", action="store", type=Path, required=True)
    parser.add_argument("--splits", action="store", type=int, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    assert str(args.src_pklgz).endswith(".pkl.gz")
    with gzip.open(args.src_pklgz, "rb") as fobj:
        segments = pickle.load(fobj)
    LOGGER.info(f"Loaded {len(segments)} segments")
    for split in range(args.splits):
        start_idx, end_idx = utils.get_start_end_idx(len(segments), args.splits, split + 1)
        fn = str(args.src_pklgz)[:-6] + f"{split}.pkl.gz"
        with gzip.open(fn, "wb") as fobj:
            pickle.dump(segments[start_idx:end_idx], fobj)
        LOGGER.info(f"Wrote {end_idx - start_idx} segs to {fn}")


if __name__ == "__main__":
    main()
