#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import kaldi_io
import torchaudio

from acn_embed.util import utils
from acn_embed.util.data import transcription
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)
THIS_DIR = Path(__file__).parent.absolute()


# pylint: disable=too-many-locals


def main():
    parser = argparse.ArgumentParser(
        description=("Extract kaldi MFCCs"), formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--src-tran", action="store", type=Path, required=True)
    parser.add_argument("--audio-path", action="store", type=Path, required=True)
    parser.add_argument("--dst-path", action="store", type=Path, required=True)
    parser.add_argument("--splits", action="store", type=int, required=True)
    parser.add_argument("--num", action="store", type=int, required=True)
    parser.add_argument("--kaldi-bin", action="store", type=str, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    num_processed = 0
    utt2spk_fn = os.path.join(args.dst_path, f"utt2spk.{args.num}")
    text_fn = os.path.join(args.dst_path, f"text.{args.num}")

    ark_fn = os.path.join(args.dst_path, f"mfcc.ark.{args.num}")
    scp_fn = os.path.join(args.dst_path, f"mfcc.scp.{args.num}")
    wspecifier = f"ark,scp:{ark_fn},{scp_fn}"

    wspec = f"ark:| {args.kaldi_bin}/copy-feats --compress=true ark:- {wspecifier}"

    total_num_utts = utils.count_lines(args.src_tran)
    start_idx, end_idx = utils.get_start_end_idx(total_num_utts, args.splits, args.num)
    LOGGER.info(f"{total_num_utts=} {start_idx=} {end_idx=}")

    with kaldi_io.open_or_fd(wspec, "wb") as fobj:
        dst_tran = []
        for num, utt_info in enumerate(transcription.read_tran_utts(args.src_tran)):

            if not start_idx <= num < end_idx:
                continue

            if num_processed % 1000 == 0:
                LOGGER.info(f"{num_processed=}")
            num_processed += 1

            waveform_fn = args.audio_path / utt_info["audio_fn"]
            waveform, sample_rate = torchaudio.load(waveform_fn)
            assert sample_rate == 16000

            # Use same default params as compute-mfcc-feats
            mfcc = (
                torchaudio.compliance.kaldi.mfcc(
                    waveform=waveform,
                    num_ceps=13,
                    sample_frequency=sample_rate,
                    use_energy=True,
                    energy_floor=0,
                    dither=1.0 / 32767.0,  # Equivalent of 1.0 in kaldi (using 16-bit wav)
                )
                .detach()
                .cpu()
                .numpy()
            )

            kaldi_io.write_mat(fobj, mfcc, key=utt_info["utt_id"])

            dst_tran.append(utt_info)

    LOGGER.info(f"{num_processed=}")

    # We only do speaker-independent training, but all the kaldi egs scripts want utt2spk
    with open(utt2spk_fn, "w", encoding="utf8") as fobj:
        for tran in dst_tran:
            fobj.write(f"{tran['utt_id']} global\n")

    with open(text_fn, "w", encoding="utf8") as fobj:
        for tran in dst_tran:
            fobj.write(f"{tran['utt_id']} {tran['text']}\n")


if __name__ == "__main__":
    main()
