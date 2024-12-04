#!/usr/bin/env python3
import argparse
import gzip
import json
import re
import sys

from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--src", action="store", type=str, required=True)
    parser.add_argument("--dst", action="store", type=str, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    rejected_chars_re = re.compile(r"[^A-Z' ]")
    word_set = set()
    with gzip.open(args.src, "rt") as fobj:
        for line in fobj:
            jsonl = json.loads(line)
            assert len(jsonl["supervisions"]) == 1
            supervision = jsonl["supervisions"][0]
            assert len(jsonl["recording"]["sources"]) == 1
            text = supervision["custom"]["texts"][1]
            for word in text.split():
                word = rejected_chars_re.sub("", word.strip()).lstrip("'")
                word_set.add(word)

    with open(args.dst, "w", encoding="utf8") as fobj:
        fobj.writelines([word + "\n" for word in sorted(word_set)])


if __name__ == "__main__":
    main()
