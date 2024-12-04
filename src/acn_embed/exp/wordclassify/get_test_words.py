#!/usr/bin/env python3

import argparse
import random
import sys
from pathlib import Path

from acn_embed.util.data.transcription import read_tran_utts, TranscriptionWriter
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


# pylint:disable=too-many-branches


def get_isolated_words(tokens, min_len_ms):
    isolated_tokens = []
    for token in tokens[:-1]:
        if token["end_ms"] - token["start_ms"] >= min_len_ms:
            isolated_tokens.append(
                {
                    "orth": token["orth"],
                    "pron": token["pron"],
                    "start_ms": token["start_ms"],
                    "end_ms": token["end_ms"],
                }
            )
    return isolated_tokens


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Get random list of isolated words for testing",
    )
    parser.add_argument("--tran", action="store", type=str, nargs="+", required=True)
    parser.add_argument("--num-words", action="store", type=int, required=True)
    parser.add_argument("--min-len-ms", action="store", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    new_utts = []
    seen_words = set()

    utts = []
    for tran in args.tran:
        for utt in read_tran_utts(tran):
            utt["src_tran"] = tran
            utts.append(utt)

    random.shuffle(utts)
    for utt in utts:
        for token in get_isolated_words(
            tokens=utt["force_align"][0]["tokens"], min_len_ms=args.min_len_ms
        ):
            if token["orth"] not in seen_words:
                new_utt = {
                    "src_tran": utt["src_tran"],
                    "src_utt_id": utt["utt_id"],
                    "audio_fn": utt["audio_fn"],
                    "ref_token": token,
                    "text": token["orth"],
                }
                new_utts.append(new_utt)
                seen_words.add(token["orth"])

    LOGGER.info(f"Found {len(new_utts)} isolated tokens")

    if args.num_words < len(new_utts):
        new_utts = random.sample(new_utts, args.num_words)

    with TranscriptionWriter(args.output) as writer:
        for test_word_num, utt in enumerate(new_utts):
            utt["utt_id"] = f"test_word_{test_word_num:05d}"
            writer.write(utt)


if __name__ == "__main__":
    main()
