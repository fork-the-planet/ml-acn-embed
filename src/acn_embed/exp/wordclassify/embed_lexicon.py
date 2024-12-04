#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path

import torch

from acn_embed.embed.embedder.text_embedder import TextEmbedder
from acn_embed.util.lexicon import Lexicon
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--subword", choices=["grapheme", "phone"], required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    embedder = TextEmbedder(args.model, text_type=args.subword)
    if args.subword == "phone":
        inputs = [pron.split() for pron in sorted(Lexicon(args.lexicon).get_pron_set())]
    elif args.subword == "grapheme":
        inputs = sorted(Lexicon(args.lexicon).get_word_set())
    else:
        raise RuntimeError("Coding error")
    emb_tensor = embedder.get_embedding(inputs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as fobj:
        pickle.dump(inputs, fobj)
        torch.save(emb_tensor, fobj, _use_new_zipfile_serialization=False)

    LOGGER.info(f"{emb_tensor.shape=}")


if __name__ == "__main__":
    main()
