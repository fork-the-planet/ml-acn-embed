#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path

import torch

from acn_embed.embed.embedder.text_embedder import TextEmbedder
from acn_embed.util.data.transcription import read_tran_utts, TranscriptionWriter
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


# pylint:disable=too-many-locals


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--subword", choices=["grapheme", "phone"], required=True)
    parser.add_argument("--oovs", type=Path, required=True)
    parser.add_argument("--asr-tran", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    embedder = TextEmbedder(args.model, text_type=args.subword, device=device)

    with args.oovs.open("rb") as fobj:
        words = pickle.load(fobj)
        g_emb = torch.load(fobj, map_location=device, weights_only=True)
        assert len(words) == g_emb.shape[0]
        LOGGER.info(f"{g_emb.shape=}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with TranscriptionWriter(args.output) as writer:
        for utt_info in read_tran_utts(args.asr_tran):
            asr_token = utt_info["asr_token"]
            if args.subword == "phone":
                if not asr_token["pron"]:
                    continue
                inputs = [asr_token["pron"].split()]
            else:
                if not asr_token["orth"]:
                    continue
                inputs = [asr_token["orth"]]
            input_emb = embedder.get_embedding(inputs).to(device)
            distances = torch.sqrt(torch.sum(torch.pow(input_emb - g_emb, 2.0), axis=1))
            best_idx = torch.argmin(distances)
            best_word = words[best_idx]
            if args.subword == "grapheme":
                hyp_token = {"orth": best_word}
            else:
                hyp_token = {"pron": " ".join(best_word)}
            writer.write(
                {
                    "utt_id": utt_info["utt_id"],
                    "ref_token": utt_info["ref_token"],
                    "asr_token": utt_info["asr_token"],
                    "hyp_token": hyp_token,
                }
            )


if __name__ == "__main__":
    main()
