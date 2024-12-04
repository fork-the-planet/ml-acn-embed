#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path

import torch

from acn_embed.embed.embedder.audio_embedder import AudioEmbedder
from acn_embed.util.data.storage import H5FeatStoreReader
from acn_embed.util.data.transcription import read_tran_utts, TranscriptionWriter
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


# pylint:disable=too-many-locals


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--cuda-device", type=int, default=-1)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--subword", choices=["grapheme", "phone"], required=True)
    parser.add_argument("--test-fbank", type=Path, required=True)
    parser.add_argument("--test-tran", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--trim-sil-thres", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    if args.cuda_device >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")

    embedder = AudioEmbedder(args.model, device=device)

    with args.search.open("rb") as fobj:
        words = pickle.load(fobj)
        g_emb = torch.load(fobj, map_location=device, weights_only=True)
        assert len(words) == g_emb.shape[0]
        LOGGER.info(f"{g_emb.shape=}")

    reader = H5FeatStoreReader(args.test_fbank)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with TranscriptionWriter(args.output) as writer:
        for num, utt_info in enumerate(read_tran_utts(args.test_tran)):

            nparr = reader.get_nparr(num)
            f_emb = embedder.get_embedding(
                fbank_tensor=torch.tensor(nparr, device=device), trim_sil_thres=args.trim_sil_thres
            )
            distances = torch.sqrt(torch.sum(torch.pow(f_emb - g_emb, 2.0), axis=1))
            best_idx = torch.argmin(distances)
            best_word = words[best_idx]
            if args.subword == "grapheme":
                token = {"orth": best_word}
            else:
                token = {"pron": " ".join(best_word)}
            utt_info["hyp_token"] = token
            writer.write(utt_info)


if __name__ == "__main__":
    main()
