#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from acn_embed.util.data import transcription
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--max-utts", action="store", type=int, default=None)
    parser.add_argument("--max-words-per-utt", action="store", type=int, default=None)
    parser.add_argument("--min-words-per-utt", action="store", type=int, default=None)
    parser.add_argument("--dst-tran", action="store", type=Path, required=True)
    parser.add_argument("--src-tran", action="store", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    with transcription.TranscriptionWriter(args.dst_tran) as writer:
        for utt in transcription.read_tran_utts(args.src_tran):
            num_words = len(utt["text"].split())
            if args.min_words_per_utt and num_words < args.min_words_per_utt:
                continue
            if args.max_words_per_utt and num_words > args.max_words_per_utt:
                continue
            writer.write(utt)
            if args.max_utts and writer.num_written >= args.max_utts:
                break


if __name__ == "__main__":
    main()
