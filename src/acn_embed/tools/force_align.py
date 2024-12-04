#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

from acn_embed.fe_am.force_aligner.force_aligner import ForceAligner
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)
THIS_DIR = Path(__file__).parent.absolute()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Force-align audio to text using a DNN-HMM acoustic model "
            "trained using acn_embed. "
            "Requires kaldi source and compiled binaries, with paths given "
            "in environment variables or command line (see help)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--g2p", type=Path, default=None)
    parser.add_argument(
        "--kaldi-bin", type=Path, default=os.getenv("KALDI_BIN", ""), help="Path to kaldi binaries"
    )
    parser.add_argument(
        "--kaldi-src", type=Path, default=os.getenv("KALDI_SRC", ""), help="Path to kaldi src"
    )
    parser.add_argument("--ref", type=str, default=None, required=True)
    parser.add_argument("--model-dir", type=Path, default=None, required=True)
    parser.add_argument("--lexicon", type=Path, default=None, required=True)
    parser.add_argument("--beam", type=float, default=10.0)
    parser.add_argument("--retry-beam", type=float, default=40.0)
    parser.add_argument("--wav", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    force_aligner = ForceAligner(
        model=args.model_dir,
        lexicon=args.lexicon,
        g2p_model=args.g2p,
        kaldi_bin_path=args.kaldi_bin,
        kaldi_src_path=args.kaldi_src,
        beam=args.beam,
        retry_beam=args.retry_beam,
        device=None,
    )

    tokens = force_aligner.force_align(words=args.ref.split(), wav_fn=args.wav)

    print(json.dumps(tokens))


if __name__ == "__main__":
    main()
