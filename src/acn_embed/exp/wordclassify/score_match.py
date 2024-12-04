#!/usr/bin/env python3
import argparse
from pathlib import Path

from acn_embed.util.data.transcription import read_tran_utts
from acn_embed.util.lexicon import Lexicon
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def is_correct_pron(utt_id: str, hyp_pron: str, ref_token: dict, lexicon: Lexicon):
    ref_orth = ref_token["orth"]
    if lexicon:
        ref_prons = lexicon.word2prons[ref_orth]
    else:
        ref_prons = [ref_token["pron"]]
    if hyp_pron in ref_prons:
        return 1
    LOGGER.debug(f"Wrong: {utt_id} {ref_orth=} " f"ref_pron={ref_token['pron']} {hyp_pron=}")
    return 0


def is_correct_orth(utt_id: str, hyp_orth: str, ref_orth: str):
    if hyp_orth == ref_orth:
        return 1
    LOGGER.debug(f"Wrong: {utt_id} {ref_orth=} {hyp_orth=}")
    return 0


def score(hyp_key, hyp_tran_list, ref_tran, lexicon, subword):
    if lexicon:
        lex = Lexicon(lexicon)
    else:
        LOGGER.info("No lexicon specified. Only exact matches will be accepted.")
        lex = None

    correct = 0
    total = 0
    missing = 0

    id_to_hyp_utt_info = {}
    for hyp_tran in hyp_tran_list:
        for utt_info in read_tran_utts(hyp_tran):
            id_to_hyp_utt_info[utt_info["utt_id"]] = utt_info

    for ref_utt in read_tran_utts(ref_tran):
        utt_id = ref_utt["utt_id"]
        if utt_id in id_to_hyp_utt_info:
            hyp_info = id_to_hyp_utt_info[utt_id]

            if subword == "phone":
                correct += is_correct_pron(
                    utt_id=utt_id,
                    hyp_pron=hyp_info[hyp_key]["pron"],
                    ref_token=ref_utt["ref_token"],
                    lexicon=lex,
                )
            else:
                correct += is_correct_orth(
                    utt_id=utt_id,
                    hyp_orth=hyp_info[hyp_key]["orth"],
                    ref_orth=ref_utt["ref_token"]["orth"],
                )
        else:
            LOGGER.warning(f"Missing hyp for {utt_id}")
            missing += 1
        total += 1

    return correct, total, missing


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--hyp-key", type=str, required=True)
    parser.add_argument("--hyp-tran", nargs="+", type=Path, required=True)
    parser.add_argument("--ref-tran", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, default=None)
    parser.add_argument("--subword", choices=["grapheme", "phone"], required=True)
    args = parser.parse_args()

    correct, total, missing = score(
        hyp_key=args.hyp_key,
        hyp_tran_list=args.hyp_tran,
        ref_tran=args.ref_tran,
        lexicon=args.lexicon,
        subword=args.subword,
    )
    print(f"{correct=} {total=} {missing=} acc={correct * 100.0 / total:.1f}")


if __name__ == "__main__":
    main()
