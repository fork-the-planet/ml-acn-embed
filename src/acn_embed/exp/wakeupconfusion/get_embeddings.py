#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch
from acn_embed.embed.embedder.text_embedder import TextEmbedder
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Get embeddings"
    )
    parser.add_argument("--batchsize", action="store", type=int, default=500)
    parser.add_argument("--str2score", action="store", type=Path, required=True)
    parser.add_argument("--model", action="store", type=Path, required=True)
    parser.add_argument("--output", action="store", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    with open(args.str2score, "rb") as fobj:
        strings = torch.load(fobj, weights_only=True, map_location="cpu")["strings"]

    LOGGER.info(f"Read {len(strings)} strings")

    embedder = TextEmbedder(model_dir=args.model, text_type="grapheme")

    embeddings = embedder.get_embedding(text=strings, batch_size=args.batchsize)

    with args.output.open("wb") as fobj:
        torch.save(embeddings, fobj)


if __name__ == "__main__":
    main()
