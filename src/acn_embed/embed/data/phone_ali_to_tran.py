#!/usr/bin/env python3
import argparse
import gzip
import sys

from acn_embed.util.data.transcription import read_tran_utts, write_tran_utts
from acn_embed.util.logger import get_logger
from acn_embed.util.phone_table import PhoneTable, phone_pd_to_pi

LOGGER = get_logger(__name__)


def read_utt2_phone_len(fobj):
    utt2pl = {}
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


def get_word_frames(phone_info, phone_table: PhoneTable):
    word_start_frame = []
    word_end_frame = []
    prons = []

    phones = []
    frame = 0
    for phone_id, num_frames in phone_info:
        phone = phone_table.id_to_pd_phone[phone_id]
        LOGGER.debug(f"{frame=} {phone=}")
        if "sil" in phone:
            pass
        elif phone.endswith("_B"):
            word_start_frame.append(frame)
            phones.append(phone)
        elif phone.endswith("_I"):
            phones.append(phone)
        elif phone.endswith("_E"):
            word_end_frame.append(frame + num_frames)
            prons.append(phones + [phone])
            phones = []
        elif phone.endswith("_S"):
            assert not phones
            word_start_frame.append(frame)
            word_end_frame.append(frame + num_frames)
            prons.append([phone])
        frame += num_frames

    assert len(word_start_frame) == len(word_end_frame) == len(prons)
    return list(zip(word_start_frame, word_end_frame, prons, strict=True))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--phones-ark", action="store", type=str, default=None)
    parser.add_argument("--phone-table", action="store", type=str, default=None)
    parser.add_argument("--tran", action="store", nargs="+", type=str, required=True)
    parser.add_argument("--output", action="store", type=str, required=True)
    parser.add_argument("--frame-rate-ms", action="store", type=int, default=10)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    phone_table = PhoneTable(args.phone_table)

    with gzip.open(args.phones_ark, "rt") as fobj:
        utt2pl = read_utt2_phone_len(fobj)

    tran_list = []

    for tran_fn in args.tran:
        for utt_info in read_tran_utts(tran_fn):
            if utt_info["utt_id"] not in utt2pl:
                continue
            words = utt_info["text"].split()
            word_frames = get_word_frames(utt2pl[utt_info["utt_id"]], phone_table)
            assert len(word_frames) == len(words)
            utt_info["force_align"] = [
                {
                    "tokens": [
                        {
                            "orth": words[idx],
                            "pron": " ".join(
                                [phone_pd_to_pi(phone) for phone in word_frames[idx][2]]
                            ),
                            "start_ms": word_frames[idx][0] * args.frame_rate_ms,
                            "end_ms": word_frames[idx][1] * args.frame_rate_ms,
                        }
                        for idx in range(len(words))
                    ]
                }
            ]
            tran_list.append(utt_info)
    write_tran_utts(tran_list, args.output)
    LOGGER.info(f"Wrote {len(tran_list)} utts to {args.output}")


if __name__ == "__main__":
    main()
