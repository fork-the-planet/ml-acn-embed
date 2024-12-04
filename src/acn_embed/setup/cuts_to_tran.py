#!/usr/bin/env python3
import argparse
import gzip
import random
import re
import sys
from pathlib import Path

from acn_embed.setup.cuts_reader import read_cuts_utts
from acn_embed.util.data.transcription import TranscriptionWriter
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)

APOSTROPHE_RE = re.compile(r"[‘’]")
DROP_CHARS_RE = re.compile(r"[^A-Z0-9' ]")


def normalize_text(text: str):
    normalized = APOSTROPHE_RE.sub(repl="'", string=text)
    normalized = DROP_CHARS_RE.sub(repl=" ", string=normalized.upper())
    return " ".join(normalized.split())


def readlines(file: Path):
    with (
        gzip.open(file, "rt", encoding="utf8")
        if file.name.endswith(".gz")
        else open(file, "r", encoding="utf8")
    ) as fobj:
        return [line.strip() for line in fobj.readlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Convert LibriHeavy cuts jsonl file to transcription file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src", action="store", type=Path, required=True)
    parser.add_argument("--dst", action="store", type=Path, required=True)
    parser.add_argument("--list", action="store", type=Path, default=None)
    parser.add_argument("--shuffle", action="store_true", default=False)
    args = parser.parse_args()

    LOGGER.info(" ".join(sys.argv))

    use_uttids = set()
    if args.list:
        use_uttids = set(readlines(args.list))
        LOGGER.info(f"Read {len(use_uttids)} utt IDs")

    norm_failed = 0
    length_s = 0
    seen_utt_id_set = set()
    dst_utts = []
    for utt in read_cuts_utts(args.src):

        utt_id = utt["utt_id"]
        if use_uttids and utt_id not in use_uttids:
            continue

        normalized = normalize_text(utt["orig_text"])

        if not normalized:
            LOGGER.info(f"Failed to normalize: {utt['orig_text']}")
            norm_failed += 1
            continue

        assert utt_id not in seen_utt_id_set
        seen_utt_id_set.add(utt_id)

        utt["audio_fn"] = utt_id + ".wav"
        utt["text"] = normalized

        dst_utts.append(utt)
        length_s += utt["length_s"]

    if args.shuffle:
        random.shuffle(dst_utts)

    with TranscriptionWriter(args.dst) as writer:
        for utt in dst_utts:
            writer.write(utt)

    written = len(dst_utts)
    total = written + norm_failed
    LOGGER.info(
        f"Wrote:{written}, Total hours:{length_s / 60 / 60:.1f}, "
        f"Failed to normalize:{norm_failed} ({100.0 * norm_failed / total:.2f}%)"
    )


if __name__ == "__main__":
    main()
