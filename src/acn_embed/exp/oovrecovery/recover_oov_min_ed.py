#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import numpy as np

from acn_embed.exp.oovrecovery.min_ed import PronMinEditDistance
from acn_embed.util.data.transcription import read_tran_utts, TranscriptionWriter
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ref-tran", type=Path, required=True)
    parser.add_argument("--asr-tran", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    oov_prons = [utt["ref_token"]["pron"].split() for utt in read_tran_utts(args.ref_tran)]

    pron_min_ed = PronMinEditDistance(
        model_dir=args.model, sigma=args.sigma, insert_cost=1, delete_cost=1
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with TranscriptionWriter(args.output) as writer:
        for num, utt_info in enumerate(read_tran_utts(args.asr_tran)):

            if args.limit and num >= args.limit:
                break

            if num % 100 == 0:
                LOGGER.info(f"{num=}")

            token = utt_info["asr_token"]
            if not token["pron"]:
                continue
            asr_pron = token["pron"].split()

            distances = [pron_min_ed.get_distance(asr_pron, oov_pron) for oov_pron in oov_prons]

            best_idx = np.argmin(distances)
            best_pron = oov_prons[best_idx]

            utt_info["hyp_token"] = {"pron": " ".join(best_pron)}

            writer.write(utt_info)


if __name__ == "__main__":
    main()
