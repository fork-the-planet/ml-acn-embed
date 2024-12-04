#!/usr/bin/env python3

"""
Convert & split all LibriHeavy .flac files to utterance .wav files
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torchaudio

import acn_embed.util.utils
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def get_wav(audio_fn, tmp_wav_fn):
    if os.path.exists(tmp_wav_fn):
        os.remove(tmp_wav_fn)

    subprocess.run(
        f"ffmpeg -y -i {audio_fn} -ar 16000 -bits_per_raw_sample 2 -ac 1 {tmp_wav_fn}",
        shell=True,
        capture_output=True,
        check=True,
    )
    waveform, sample_rate = torchaudio.load(tmp_wav_fn)
    assert sample_rate == 16000
    return waveform


def count_lines(jsonl_file):
    count = 0
    with gzip.open(jsonl_file, "rt", encoding="utf8") as fobj:
        for _ in fobj:
            count += 1
    return count


# pylint:disable=too-many-locals
def main():
    parser = argparse.ArgumentParser(
        description=("Write utterance-wise wav files"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--libriheavy", action="store", type=Path, required=True)
    parser.add_argument("--data-type", action="store", type=str, required=True)
    parser.add_argument("--splits", action="store", type=int, required=True)
    parser.add_argument("--split-num", action="store", type=int, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    _, tmp_wav_fn = tempfile.mkstemp(suffix=f"{args.split_num}.wav")
    LOGGER.info(f"temp {tmp_wav_fn=}")

    jsonl_file = args.libriheavy / f"libriheavy_cuts_{args.data_type}.jsonl.gz"
    num_lines = count_lines(jsonl_file)

    start_idx, end_idx = acn_embed.util.utils.get_start_end_idx(
        total=num_lines, num_splits=args.splits, split=args.split_num
    )

    num_processed = 0
    utt_id_set = set()
    prev_audio_fn = None
    waveform = None

    with gzip.open(jsonl_file, "rt", encoding="utf8") as fobj:
        for line_num, line in enumerate(fobj):

            if not start_idx <= line_num < end_idx:
                continue

            jsonl = json.loads(line)
            assert len(jsonl["supervisions"]) == 1
            assert len(jsonl["recording"]["sources"]) == 1
            audio_fn = args.libriheavy / jsonl["recording"]["sources"][0]["source"]
            orig_wav_end_s = jsonl["start"] + jsonl["duration"]
            orig_wav_start_s = jsonl["start"]
            utt_id = jsonl["id"]

            # Sanity check
            assert utt_id not in utt_id_set
            utt_id_set.add(utt_id)

            if audio_fn != prev_audio_fn:
                waveform = get_wav(audio_fn, tmp_wav_fn)
                prev_audio_fn = audio_fn

            wave_segment = waveform[:, int(orig_wav_start_s * 16000) : int(orig_wav_end_s * 16000)]

            wave_file = args.libriheavy / "utt_wav" / (utt_id + ".wav")
            wave_file.parent.mkdir(parents=True, exist_ok=True)

            torchaudio.save(
                wave_file,
                wave_segment,
                sample_rate=16000,
                channels_first=True,
                format="wav",
                encoding="PCM_S",
                bits_per_sample=16,
            )
            if num_processed % 500 == 0:
                LOGGER.info(f"audio {num_processed=} {line_num=} {wave_file=}")
            num_processed += 1

    LOGGER.info(f"Total processed = {num_processed}")


if __name__ == "__main__":
    main()
