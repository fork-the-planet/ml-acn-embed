#!/usr/bin/env python3

import argparse
import random
import re
import sys
from pathlib import Path

from acn_embed.util.data.transcription import read_tran_utts
from acn_embed.util.lexicon import Lexicon, get_merged_lexicon
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)

DROP_CHARS_RE = re.compile(r"[^A-Z' ]")


def sanitize_text(text: str):
    normalized = DROP_CHARS_RE.sub(repl=" ", string=text.upper())
    return " ".join(normalized.split())


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Create vocab"
    )
    parser.add_argument("--word-list", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, nargs="+", required=True)
    parser.add_argument("--test-words-tran", type=Path, required=True)
    parser.add_argument("--size", action="store", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    LOGGER.info(" ".join(sys.argv))

    lex = get_merged_lexicon(args.lexicon)
    new_lex = Lexicon(word2prons={})

    test_words = set()
    for utt_info in read_tran_utts(args.test_words_tran):
        test_word = utt_info["ref_token"]["orth"]
        assert test_word in lex.word2prons
        for pron in lex.word2prons[test_word]:
            new_lex.add_pron(test_word, pron)
        test_words.add(test_word)

    other_words = set()
    with open(args.word_list, "r", encoding="utf8") as fobj:
        for word in fobj.readlines():
            word = sanitize_text(word.strip())
            if not word:
                continue
            other_words.add(word)
    other_words -= test_words
    random.shuffle(list(other_words))

    chosen_words = list(test_words)

    while len(chosen_words) < args.size:
        word = other_words.pop()
        if word not in lex.word2prons:
            continue
        for pron in lex.word2prons[word]:
            new_lex.add_pron(word, pron)
        chosen_words.append(word)

    new_lex.write(args.output)


if __name__ == "__main__":
    main()
