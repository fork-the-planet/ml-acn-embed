#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

from acn_embed.embed.embedder.audio_embedder import AudioEmbedder
from acn_embed.util.data.transcription import read_tran_utts
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def get_dr_to_f_embeddings(
    align_tran: Path, search_wav_fn: str, audio_embedder: AudioEmbedder, dr_to_f_embeddings: dict
):
    ref_text = None
    for utt in read_tran_utts(align_tran):
        wav_file = Path(utt["audio_fn"])
        if wav_file.name != search_wav_fn:
            continue
        dr_code = wav_file.parents[1].name
        LOGGER.info(f"{wav_file=} {dr_code=}")
        segments = [
            [token["start_ms"], token["end_ms"]] for token in utt["force_align"][0]["tokens"]
        ]
        f_embeddings = audio_embedder.get_embedding(wav_fn=wav_file, segments=np.array(segments))
        if dr_code not in dr_to_f_embeddings:
            dr_to_f_embeddings[dr_code] = []
        dr_to_f_embeddings[dr_code].append(f_embeddings)
        if ref_text is None:
            ref_text = utt["text"]
        # The text should be the same throughout
        assert ref_text == utt["text"]
    return ref_text


def main():
    parser = argparse.ArgumentParser(
        description="Get F embeddings from TIMIT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--f-model", type=Path, default=None, required=True)
    parser.add_argument("--search-wav", type=str, default=None, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tran", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    audio_embedder = AudioEmbedder(model_dir=args.f_model, device=None)
    dr_to_f_embeddings = {}
    ref_text = get_dr_to_f_embeddings(
        align_tran=args.tran,
        search_wav_fn=args.search_wav,
        audio_embedder=audio_embedder,
        dr_to_f_embeddings=dr_to_f_embeddings,
    )
    with open(args.output, "wb") as fobj:
        pickle.dump(ref_text, fobj)
        pickle.dump(dr_to_f_embeddings, fobj)


if __name__ == "__main__":
    main()
