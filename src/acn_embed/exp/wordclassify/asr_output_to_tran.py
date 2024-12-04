#!/usr/bin/env python3
import argparse
import gzip
import sys
from pathlib import Path

from acn_embed.util.data.transcription import read_tran_utts, write_tran_utts
from acn_embed.util.phone_table import PhoneTable, phone_pd_to_pi
from acn_embed.util.word_table import WordTable
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def read_utt_to_phone_len(ark_fn):
    utt2pl = {}
    with gzip.open(ark_fn, "rt") as fobj:
        for line in fobj:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            uttid = tokens[0]
            phone_info = []
            for seg in " ".join(tokens[1:]).split(";"):
                phone_id, num_frames = [int(x) for x in seg.strip().split()]
                phone_info.append([phone_id, num_frames])
            assert uttid not in utt2pl
            utt2pl[uttid] = phone_info
    return utt2pl


def iter_read(fn):
    with (
        gzip.open(fn, "rt") if str(fn).endswith(".gz") else open(fn, "r", encoding="utf8")
    ) as fobj:
        for line in fobj:
            line = line.strip()
            if line:
                yield line.split()


def read_utt_to_words(ark_fn, word_table: WordTable):
    utt2str = {}
    for tokens in iter_read(ark_fn):
        uttid = tokens[0]
        wordids = [int(tok) for tok in tokens[1:]]
        words = [word_table.id_to_word[id] for id in wordids]
        utt2str[uttid] = " ".join(words)
    return utt2str


def get_pron(phone_info, phone_table: PhoneTable):
    phones = []
    for phone_id, _ in phone_info:
        phone = phone_pd_to_pi(phone_table.id_to_pd_phone[phone_id])
        if "sil" not in phone:
            phones.append(phone)
    return " ".join(phones)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--words-ark", action="store", type=Path, default=None)
    parser.add_argument("--word-table", action="store", type=Path, required=True)
    parser.add_argument("--phones-ark", action="store", type=Path, default=None)
    parser.add_argument("--phone-table", action="store", type=Path, default=None)
    parser.add_argument("--tran", action="store", nargs="+", type=Path, required=True)
    parser.add_argument("--output", action="store", type=Path, required=True)
    parser.add_argument("--frame-rate-ms", action="store", type=int, default=10)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    phone_table = PhoneTable(args.phone_table)
    word_table = WordTable(args.word_table)

    utt2pl = read_utt_to_phone_len(args.phones_ark)
    utt2str = read_utt_to_words(args.words_ark, word_table)

    tran_list = []

    for tran_fn in args.tran:
        for utt_info in read_tran_utts(tran_fn):
            words = utt_info["text"].split()
            assert len(words) == 1
            utt_id = utt_info["utt_id"]
            if utt_id in utt2pl:
                pron = get_pron(utt2pl[utt_id], phone_table)
            else:
                pron = ""
            utt_info["asr_token"] = {"orth": utt2str.get(utt_id, ""), "pron": pron}
            tran_list.append(utt_info)
    write_tran_utts(tran_list, args.output)
    LOGGER.info(f"Wrote {len(tran_list)} utts to {args.output}")


if __name__ == "__main__":
    main()
