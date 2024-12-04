#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from acn_embed.embed.embedder.text_embedder import TextEmbedder
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def read_batch_jsonl(path: Path, batch_size: int):
    batch = []
    with path.open("r", encoding="utf8") as fobj:
        for line in fobj:
            line = line.strip()
            if line:
                dic = json.loads(line.strip())
                batch.append(dic)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(
        description="Run phone embedder", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model", type=Path, help="Model directory")
    parser.add_argument(
        "--pron", type=str, default=None, help='Phone sequence, e.g. "N UW1 Y AO1 R K"'
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
            '"id" and "pron"'
        ),
    )
    parser.add_argument(
        "--output-jsonl", type=Path, default=None, help="Bulk processing output file."
    )
    args = parser.parse_args()

    embedder = TextEmbedder(model_dir=args.model, text_type="phone")

    if args.pron:
        assert (
            args.input_jsonl is None and args.output_jsonl is None
        ), "You cannot supply --input-jsonl or --output-jsonl if --text is specified"
        vector = embedder.get_embedding([args.pron.split()]).cpu().numpy()
        np.set_printoptions(precision=4, linewidth=100)
        print(vector)
    else:
        assert (
            args.input_jsonl and args.output_jsonl
        ), "If you don't supply --pron, you must supply --input-jsonl and --output-jsonl"
        with args.output_jsonl.open("w", encoding="utf8") as write_obj:
            for batch in read_batch_jsonl(args.input_jsonl, batch_size=args.batch_size):
                vectors = (
                    embedder.get_embedding(
                        [obj["pron"].split() for obj in batch], batch_size=len(batch)
                    )
                    .cpu()
                    .tolist()
                )
                for vector, obj in zip(vectors, batch):
                    json.dump(
                        {"id": obj["id"], "pron": obj["pron"], "embedding": vector}, write_obj
                    )
                    write_obj.write("\n")


if __name__ == "__main__":
    main()
