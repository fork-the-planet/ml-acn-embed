#!/usr/bin/env python3

import argparse
import random
import sys
from pathlib import Path

from acn_embed.exp.wordclassify.create_vocab import sanitize_text
from acn_embed.util.data.transcription import read_tran_utts
from acn_embed.util.lexicon import Lexicon, get_merged_lexicon
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create a deficient lexicon",
    )
    parser.add_argument("--lexicon", type=Path, nargs="+", required=True)
    parser.add_argument("--test-words-tran", type=Path, required=True)
    parser.add_argument("--size", action="store", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    lex = get_merged_lexicon(args.lexicon)
    deficient_lex = Lexicon(word2prons={})

    test_prons = set()
    test_words = set()
    for utt_info in read_tran_utts(args.test_words_tran):
        test_prons.add(utt_info["ref_token"]["pron"])
        test_words.add(utt_info["ref_token"]["orth"])

    # Remove test prons from the lexicon
    shuffled_words = list({sanitize_text(word) for word in lex.word2prons})
    random.shuffle(shuffled_words)
    for word in shuffled_words:
        if args.size > 0 and len(deficient_lex.word2prons) >= args.size:
            break
        if (word in test_words) or (word not in lex.word2prons):
            continue
        for pron in lex.word2prons[word]:
            if pron not in test_prons:
                deficient_lex.add_pron(word, pron)

    deficient_lex.write(args.output)


if __name__ == "__main__":
    main()
