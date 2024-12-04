#!/usr/bin/env python3
import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from acn_embed.util.lexicon import Lexicon
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def create_grammar_fst(words, output_file):
    _, tmpfn = tempfile.mkstemp()
    LOGGER.info(f"{tmpfn=}")

    # Same weight for every word
    weight = -math.log(1.0 / len(words))

    start_state = 0
    end_state = 1
    with open(tmpfn, "w", encoding="utf8") as fobj:
        for word_id0 in range(len(words)):
            word_id = word_id0 + 1
            fobj.write(f"{start_state} {end_state} {word_id} {word_id} {weight}\n")
        fobj.write(f"{end_state}\n")

    subprocess.run(
        f"fstcompile {tmpfn} | fsttopsort | fstdeterminize | fstrmepsilon "
        f"| fstminimizeencoded | fstarcsort > {output_file}",
        shell=True,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=("Adds missing prons to a lexicon"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src-lexicon", type=Path, required=True)
    parser.add_argument("--output-fst", type=Path, required=True)
    parser.add_argument("--output-words", action="store", required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    lex = Lexicon(args.src_lexicon)

    words = sorted(lex.word2prons.keys())

    with open(args.output_words, "w", encoding="utf8") as fobj:
        fobj.write("<eps> 0\n")
        for num, word in enumerate(words):
            word_id = num + 1
            fobj.write(f"{word} {word_id}\n")

    create_grammar_fst(words, args.output_fst)


if __name__ == "__main__":
    main()
