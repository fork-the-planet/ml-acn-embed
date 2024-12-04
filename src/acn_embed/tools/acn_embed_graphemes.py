#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from acn_embed.embed.embedder.text_embedder import TextEmbedder
from acn_embed.tools.acn_embed_phones import read_batch_jsonl
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run grapheme embedder", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model", type=Path, help="Model directory")
    parser.add_argument(
        "--orth",
        type=str,
        default=None,
        help='Grapheme sequence, may include spaces, e.g. "NEW YORK"',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of sequences to process simultaneously. Reduce if you run out of memory",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help=(
            '"JSON Lines" file for bulk processing. Each line should be a dictionary with keys '
            '"id" and "orth"'
        ),
    )
    parser.add_argument(
        "--output-jsonl", type=Path, default=None, help="Bulk processing output file."
    )
    args = parser.parse_args()

    embedder = TextEmbedder(model_dir=args.model, text_type="grapheme")

    if args.orth:
        assert (
            args.input_jsonl is None and args.output_jsonl is None
        ), "You cannot supply --input-jsonl or --output-jsonl if --orth is specified"
        vector = embedder.get_embedding([args.orth.upper()]).cpu().numpy()
        np.set_printoptions(precision=4, linewidth=100)
        print(vector)
    else:
        assert (
            args.input_jsonl and args.output_jsonl
        ), "If you don't supply --orth, you must supply --input-jsonl and --output-jsonl"
        with args.output_jsonl.open("w", encoding="utf8") as write_obj:
            for batch in read_batch_jsonl(args.input_jsonl, batch_size=args.batch_size):
                vectors = (
                    embedder.get_embedding(
                        [obj["orth"].upper() for obj in batch], batch_size=len(batch)
                    )
                    .cpu()
                    .tolist()
                )
                for vector, obj in zip(vectors, batch):
                    json.dump(
                        {"id": obj["id"], "orth": obj["orth"], "embedding": vector}, write_obj
                    )
                    write_obj.write("\n")


if __name__ == "__main__":
    main()
