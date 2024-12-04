#!/usr/bin/env python3
import argparse
import sys

from acn_embed.util.data.transcription import read_tran_utts
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=("Extracts 'text' file from tran file"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tran", action="store", type=str, nargs="+", required=True)
    parser.add_argument("--output", action="store", type=str, required=True)
    parser.add_argument("--sort-utt-ids", action="store_true", default=False)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    tran = []
    for tran_fn in args.tran:
        tran += list(read_tran_utts(tran_fn))

    if args.sort_utt_ids:
        tran.sort(key=lambda x: x["utt_id"])

    with open(args.output, "w", encoding="utf8") as fobj:
        for _tran in tran:
            fobj.write(f"{_tran['utt_id']} {_tran['text']}\n")


if __name__ == "__main__":
    main()
