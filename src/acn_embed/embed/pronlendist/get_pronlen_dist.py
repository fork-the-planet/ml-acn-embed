#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys
from collections import defaultdict, OrderedDict

import numpy as np
import torch

from acn_embed.embed.pronlendist.comb_iterator import CombIterator
from acn_embed.util.lexicon import Lexicon
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)
THIS_DIR = pathlib.Path(__file__).parent.absolute()


def get_prons(string, lex: Lexicon):
    words = string.split()
    if len(words) == 1:
        return lex.word2prons.get(words[0], [])
    prons = set()
    for pron in CombIterator(
        [list(lex.word2prons[word]) for word in string.split() if word in lex.word2prons]
    ):
        prons.add(" ".join(pron))
        if len(prons) >= 3:
            break
    return sorted(prons)


def main():
    parser = argparse.ArgumentParser(
        description=("Get pron length distribution"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--string-fn", action="store", required=True)
    parser.add_argument("--src-lexicon", action="store", required=True)
    parser.add_argument("--output", action="store", required=True)
    parser.add_argument("--max-words", action="store", type=int, default=None)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    lex = Lexicon(args.src_lexicon)

    LOGGER.info("Loading")

    with open(args.string_fn, "rb") as fobj:
        obj = torch.load(fobj, weights_only=True, map_location="cpu")
        strings = obj["strings"]
        scores = np.power(10.0, np.array(obj["scores"])).tolist()

    assert len(strings) == len(scores)

    string2score = dict(zip(strings, scores))

    LOGGER.info("Getting prons")

    string2prons = {}
    for string in string2score:
        if (args.max_words is not None) and (len(string.split()) > args.max_words):
            continue
        string2prons[string] = get_prons(string, lex)

    LOGGER.info("Computing distribution")

    pronlen2score = defaultdict(float)
    for string, prons in string2prons.items():
        for pron in prons:
            pronlen2score[len(pron.split())] += (1.0 / len(prons)) * string2score[string]

    # Normalize
    score_sum = sum(pronlen2score.values())
    for pronlen in pronlen2score.keys():
        pronlen2score[pronlen] /= score_sum

    for pronlen in sorted(pronlen2score.keys()):
        print(f"{pronlen}: {pronlen2score[pronlen]:.3f}")

    with open(args.output, "w", encoding="utf8") as fobj:
        json.dump(OrderedDict(sorted(pronlen2score.items())), fobj, indent=4)


if __name__ == "__main__":
    main()
