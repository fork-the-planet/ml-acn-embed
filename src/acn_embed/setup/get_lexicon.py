#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys
from pathlib import Path

from acn_embed.util.lexicon import Lexicon, get_merged_lexicon, get_lexicon_from_g2p, fix_lexicon
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)
THIS_DIR = pathlib.Path(__file__).parent.absolute()


def get_lexicon_and_oovs(words, lex: Lexicon):
    new_lex = Lexicon()
    oovs = set()
    for word in words:
        if word in lex.word2prons:
            for pron in lex.word2prons[word]:
                new_lex.add_pron(word, pron)
        else:
            oovs.add(word)
    return new_lex, oovs


def main():
    parser = argparse.ArgumentParser(
        description=("Get pronunciation lexicon"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dst-lexicon", action="store", type=Path, required=True)
    parser.add_argument("--src-lexicon", action="store", type=Path, nargs="*", required=True)
    parser.add_argument("--g2p", action="store", type=Path, required=True)
    parser.add_argument("--list", action="store", nargs="+", type=Path, required=True)
    parser.add_argument("--phones", action="store", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    word_set = set()
    for list_path in args.list:
        word_set |= set(x.strip() for x in list_path.open("r").readlines() if x.strip())

    base_lexicon = get_merged_lexicon(args.src_lexicon)

    lexicon, oovs = get_lexicon_and_oovs(word_set, base_lexicon)
    add_lex = get_lexicon_from_g2p(oovs, args.g2p)
    lexicon.merge_in(add_lex)
    lexicon = fix_lexicon(lexicon, json.load(args.phones.open("r")))

    lexicon.write(args.dst_lexicon)


if __name__ == "__main__":
    main()
