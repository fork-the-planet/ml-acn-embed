#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

from acn_embed.util.data.transcription import read_tran_utts
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)

DROP_CHARS_RE = re.compile(r"[^A-Z' ]")


def sanitize_text(text: str):
    return DROP_CHARS_RE.sub(repl="", string=text.upper())


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--src", action="store", nargs="+", type=Path, required=True)
    parser.add_argument("--dst", action="store", type=Path, required=True)
    parser.add_argument("--sanitize", action="store_true")
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    word_set = set()
    for src in args.src:
        for utt in read_tran_utts(src):
            for word in utt["text"].split():
                if args.sanitize:
                    word = sanitize_text(word.strip())
                word_set.add(word.strip())

    with args.dst.open("w", encoding="utf8") as fobj:
        fobj.writelines([word + "\n" for word in sorted(word_set)])


if __name__ == "__main__":
    main()
