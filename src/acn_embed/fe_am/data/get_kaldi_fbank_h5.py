#!/usr/bin/env python3
import argparse
import pathlib
import sys
from pathlib import Path

import torchaudio

from acn_embed.util import utils
from acn_embed.util.data import transcription
from acn_embed.util.data.storage import H5FeatStoreWriter
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)
THIS_DIR = pathlib.Path(__file__).parent.absolute()


# pylint: disable=too-many-locals


def main():
    parser = argparse.ArgumentParser(
        description=("Extract kaldi fbanks"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src-tran", type=Path, required=True)
    parser.add_argument("--audio-path", type=Path, required=True)
    parser.add_argument("--dst-path", type=Path, required=True)
    parser.add_argument("--splits", action="store", type=int, required=True)
    parser.add_argument(
        "--num",
        action="store",
        type=int,
        required=True,
        help="Number must be 1 <= number <= splits",
    )
    parser.add_argument("--snip-ref-token", action="store_true", default=False)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    total_num_utts = utils.count_lines(args.src_tran)
    start_idx, end_idx = utils.get_start_end_idx(total_num_utts, args.splits, args.num)
    LOGGER.info(f"{total_num_utts=} {start_idx=} {end_idx=}")

    num_processed = 0
    utt2spk_f = args.dst_path / f"utt2spk.{args.num}"
    text_f = args.dst_path / f"text.{args.num}"

    h5_f = args.dst_path / f"fbank.{args.num - 1}.h5"
    feat_writer = H5FeatStoreWriter(h5_f, compress=True)

    trans_fn = args.dst_path / f"tran.{args.num - 1}.jsonl.gz"
    with transcription.TranscriptionWriter(trans_fn) as trans_writer:

        uttids = []
        trans = []

        for num, utt_info in enumerate(transcription.read_tran_utts(args.src_tran)):

            if not start_idx <= num < end_idx:
                continue

            waveform_fn = args.audio_path / utt_info["audio_fn"]
            waveform, sample_rate = torchaudio.load(waveform_fn)
            assert sample_rate == 16000

            # Use same default params as compute-fbank-feats
            fbank = (
                torchaudio.compliance.kaldi.fbank(
                    waveform=waveform,
                    sample_frequency=sample_rate,
                    num_mel_bins=80,
                    dither=1.0 / 32767.0,  # Equivalent of 1.0 in kaldi (using 16-bit wav)
                )
                .detach()
                .cpu()
                .numpy()
            )

            if args.snip_ref_token:
                fbank = fbank[
                    int(utt_info["ref_token"]["start_ms"] / 10) : int(
                        utt_info["ref_token"]["end_ms"] / 10
                    ),
                    :,
                ]

            if num_processed % 1000 == 0:
                LOGGER.info(f"{num_processed=}")

            feat_writer.add_nparr(num=num_processed, nparr=fbank)

            utt_info.update({"h5_num": num_processed, "h5_file_idx": args.num - 1})
            trans_writer.write(utt_info)

            trans.append(utt_info["text"])

            uttids.append(utt_info["utt_id"])
            num_processed += 1

    feat_writer.close()
    LOGGER.info(f"{num_processed=}")

    with utt2spk_f.open("w", encoding="utf8") as fobj:
        for uttid in uttids:
            fobj.write(f"{uttid} global\n")

    with text_f.open("w", encoding="utf8") as fobj:
        for uttid, tran in zip(uttids, trans, strict=True):
            fobj.write(f"{uttid} {tran}\n")


if __name__ == "__main__":
    main()
