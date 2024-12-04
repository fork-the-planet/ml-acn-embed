#!/usr/bin/env python3

import argparse
import sys

import arpa
import torch

from acn_embed.util.logger import get_logger
from acn_embed.util.read_arpa import get_ngrams

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Get joint LM scores"
    )
    parser.add_argument("--arpa", action="store", type=str, required=True)
    parser.add_argument("--output", action="store", type=str, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    strings = sorted(get_ngrams(args.arpa))

    LOGGER.info(f"Read {len(strings)} n-grams")

    lm = arpa.loadf(args.arpa)[0]

    LOGGER.info("Loaded LM")

    scores = []
    for num, ngram in enumerate(strings):
        joint = lm.log_s(ngram, sos=False, eos=False)
        scores.append(joint)
        if num % 1000000 == 0:
            LOGGER.info(f"{num=} {ngram} {joint}")

    idx = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    strings = [strings[_idx] for _idx in idx]
    scores = [scores[_idx] for _idx in idx]

    with open(args.output, "wb") as fobj:
        torch.save({"strings": strings, "scores": scores}, fobj)


if __name__ == "__main__":
    main()
