#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import arpa

from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Get a list of all the words in an ARPA LM",
    )
    parser.add_argument("--arpa", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    model = arpa.loadf(args.arpa)[0]
    vocab = sorted(model.vocabulary())

    with args.dst.open("w", encoding="utf8") as fobj:
        fobj.writelines(
            [x.strip() + "\n" for x in vocab if x.lower() not in ["<s>", "</s>", "<unk>"]]
        )


if __name__ == "__main__":
    main()
