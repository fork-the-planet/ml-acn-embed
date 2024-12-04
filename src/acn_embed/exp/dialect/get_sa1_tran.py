#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from acn_embed.fe_am.force_aligner.force_aligner import ForceAligner
from acn_embed.util.data.transcription import TranscriptionWriter
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run force alignment and get .tran file for TIMIT SA1 utts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kaldi-bin", type=Path, default=os.getenv("KALDI_BIN", ""), help="Path to kaldi binaries"
    )
    parser.add_argument(
        "--kaldi-src", type=Path, default=os.getenv("KALDI_SRC", ""), help="Path to kaldi src"
    )
    parser.add_argument("--lexicon", type=Path, default=None, required=True)
    parser.add_argument("--model-path", type=Path, default=None, required=True)
    parser.add_argument("--output", type=str, default=None, required=True)
    parser.add_argument("--timit-path", type=Path, default=None, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    force_aligner = ForceAligner(
        model=args.model_path,
        lexicon=args.lexicon,
        g2p_model=None,
        kaldi_bin_path=args.kaldi_bin,
        kaldi_src_path=args.kaldi_src,
        beam=10,
        retry_beam=40,
        device=None,
    )

    with TranscriptionWriter(args.output) as writer:
        for root, _, files in os.walk(args.timit_path):
            for file in files:
                wav_file = Path(root) / Path(file)
                if wav_file.name.lower() != "sa1.wav":
                    continue
                txt_file = str(wav_file)[:-3] + "txt"
                with open(txt_file, "r", encoding="utf8") as fobj:
                    words = fobj.read().strip().split()
                transcription = " ".join(
                    words[2:]
                ).upper()  # skip first two tokens, which are timestamps
                if transcription.endswith("."):
                    transcription = transcription[:-1]
                assert transcription == "SHE HAD YOUR DARK SUIT IN GREASY WASH WATER ALL YEAR"
                utt_id = str(wav_file)[:-4]
                utt_id = utt_id[len(str(wav_file.parents[3])) + 1 :]
                tokens = force_aligner.force_align(words=transcription.split(), wav_fn=wav_file)
                utt = {
                    "type": "utt",
                    "utt_id": utt_id,
                    "audio_fn": str(wav_file),
                    "text": transcription,
                    "force_align": [{"tokens": tokens}],
                }
                writer.write(utt)


if __name__ == "__main__":
    main()
