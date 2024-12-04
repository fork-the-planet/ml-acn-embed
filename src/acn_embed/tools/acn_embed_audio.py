#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from acn_embed.embed.embedder.audio_embedder import AudioEmbedder
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf8") as fobj:
        for line in fobj:
            line = line.strip()
            if line:
                dic = json.loads(line.strip())
                yield dic


def main():
    parser = argparse.ArgumentParser(
        description="Run audio embedder", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model", type=Path, help="Model directory")
    parser.add_argument("--wav", type=Path, action="store", help=".wav filename")
    parser.add_argument(
        "--no-dither",
        action="store_true",
        help="Turn off dithering (adding small random noise to avoid zeros in the audio)",
    )
    parser.add_argument(
        "--start-ms",
        type=int,
        default=None,
        help="Segment start time (ms). If not specified, the entire wav file will be used.",
    )
    parser.add_argument(
        "--end-ms",
        type=int,
        default=None,
        help="Segment end time (ms). If not specified, the entire wav file will be used.",
    )
    parser.add_argument(
        "--trim-sil-thres",
        type=float,
        default=-0.1,
        help=(
            "Threshold (must be <=0) used to trim the front and back of each segment so that "
            "contiguous silence frames at the beginning and end are dropped. The lower the value, "
            "the more aggressive the trimming. The maximum value (0) results in no trimming. "
            "The default is -0.1."
        ),
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help=(
            '"JSON Lines" file for bulk processing. Each line should be a dictionary with keys '
            '"id", "wav", and "segments"'
        ),
    )
    parser.add_argument(
        "--output-jsonl", type=Path, default=None, help="Bulk processing output file."
    )

    args = parser.parse_args()

    embedder = AudioEmbedder(model_dir=args.model)

    if args.wav:
        assert (
            args.input_jsonl is None and args.output_jsonl is None
        ), "You cannot supply --input-jsonl or --output-jsonl if --wav is specified"
        if args.start_ms is None or args.end_ms is None:
            segments = None
        else:
            segments = np.array([[args.start_ms, args.end_ms]], dtype=np.int32)
        vector = (
            embedder.get_embedding(
                wav_fn=args.wav,
                segments=segments,
                trim_sil_thres=args.trim_sil_thres,
                no_dither=args.no_dither,
            )
            .detach()
            .cpu()
            .numpy()
        )
        np.set_printoptions(precision=4, linewidth=100)
        print(vector)
    else:
        assert (
            args.start_ms is None and args.end_ms is None
        ), "You cannot supply --start-ms or --end-ms if --input-jsonl is specified"
        assert (
            args.input_jsonl and args.output_jsonl
        ), "If you don't supply --wav, you must supply --input-jsonl and --output-jsonl"
        with args.output_jsonl.open("w", encoding="utf8") as write_obj:
            for obj in read_jsonl(args.input_jsonl):
                wav_path = Path(obj["wav"])
                if not wav_path.is_absolute():
                    wav_path = args.input_jsonl.parent / wav_path
                vectors = (
                    embedder.get_embedding(
                        wav_fn=wav_path,
                        segments=np.array(obj["segments"]),
                        trim_sil_thres=args.trim_sil_thres,
                        no_dither=args.no_dither,
                    )
                    .detach()
                    .cpu()
                    .tolist()
                )
                json.dump({"id": obj["id"], "embeddings": vectors}, write_obj)
                write_obj.write("\n")


if __name__ == "__main__":
    main()
