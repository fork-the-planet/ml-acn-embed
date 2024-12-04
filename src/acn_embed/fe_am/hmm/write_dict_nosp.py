#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

from acn_embed.util.lexicon import Lexicon
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)
THIS_DIR = Path(__file__).parent.absolute()


def write_dict_nosp(lexicon: Path, non_sil_phones: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(non_sil_phones, "r", encoding="utf8") as fobj:
        non_sil_phones_set = set(json.load(fobj))

    lexicon = Lexicon(path=lexicon)
    lexicon.word2prons["<UNK>"] = {"sil"}

    # lexicon.txt
    lexicon_fn = output_dir / "lexicon.txt"
    lexicon.write(lexicon_fn)

    # extra_questions.txt
    extra_questions_fn = output_dir / "extra_questions.txt"
    with open(extra_questions_fn, "w", encoding="utf8") as fobj:
        fobj.write("sil\n")
    LOGGER.debug(f"Wrote {extra_questions_fn}")

    # nonsilence_phones.txt
    nonsilence_phones_fn = output_dir / "nonsilence_phones.txt"
    with open(nonsilence_phones_fn, "w", encoding="utf8") as fobj:
        for ph in sorted(non_sil_phones_set):
            assert ph != "sil"
            fobj.write(f"{ph}\n")
    LOGGER.debug(f"Wrote {nonsilence_phones_fn}")

    # optional_silence.txt
    optional_silence_fn = output_dir / "optional_silence.txt"
    with open(optional_silence_fn, "w", encoding="utf8") as fobj:
        fobj.write("sil\n")
    LOGGER.debug(f"Wrote {optional_silence_fn}")

    # silence_phones.txt
    silence_phones_fn = output_dir / "silence_phones.txt"
    with open(silence_phones_fn, "w", encoding="utf8") as fobj:
        fobj.write("sil\n")
    LOGGER.debug(f"Wrote {silence_phones_fn}")


def main():
    parser = argparse.ArgumentParser(
        description=("Writes a dict_nosp dir for consumption by kaldi utils/prepare_lang.sh"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--non-sil-phones", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    write_dict_nosp(
        lexicon=args.lexicon, non_sil_phones=args.non_sil_phones, output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
