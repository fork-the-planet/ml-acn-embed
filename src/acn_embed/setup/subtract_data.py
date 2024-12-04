#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from acn_embed.util.data import transcription
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--a-tran", action="store", type=Path, required=True)
    parser.add_argument("--b-tran", action="store", type=Path, required=True)
    parser.add_argument("--dst", action="store", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    subtract_uttids = set()
    for utt in transcription.read_tran_utts(args.b_tran):
        subtract_uttids.add(utt["utt_id"])

    with transcription.TranscriptionWriter(args.dst) as writer:
        for utt in transcription.read_tran_utts(args.a_tran):
            if utt["utt_id"] in subtract_uttids:
                continue
            writer.write(utt)


if __name__ == "__main__":
    main()
