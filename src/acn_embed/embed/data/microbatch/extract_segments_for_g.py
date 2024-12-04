#!/usr/bin/env python3

import argparse
import gzip
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from acn_embed.embed.data.microbatch.long_segment import LongSegSpan
from acn_embed.embed.data.microbatch.lss_index import LSSIndex
from acn_embed.embed.data.microbatch.span_info import SpanInfo
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class RandomSegmentExtractor:
    def __init__(
        self, long_segment_metadata_fn: Path, index_fn: Path, pron_len_dist: Path, max_tot_segments
    ):

        with gzip.open(long_segment_metadata_fn, "rb") as fobj:
            self.long_segments = pickle.load(fobj)

        with gzip.open(index_fn, "rb") as fobj:
            self.index = LSSIndex.from_fobj(fobj)

        self.len_to_span_info = {}
        for long_segment in self.long_segments:
            if long_segment.length not in self.len_to_span_info:
                self.len_to_span_info[long_segment.length] = SpanInfo(long_segment.length)

        with open(pron_len_dist, "r", encoding="utf8") as fobj:
            num_phones_to_prior = {int(key): float(val) for key, val in json.load(fobj).items()}

        assert isinstance(num_phones_to_prior, dict)
        max_len_pron = min(max(num_phones_to_prior.keys()), self.index.max_pron_len)

        # Normalize num_phones_to_prior so that priors sum to 1
        tot_prior = np.sum(list(num_phones_to_prior.values()))
        num_phones_to_prior = {
            length: (prior / tot_prior) for (length, prior) in num_phones_to_prior.items()
        }

        self.pron_lengths = []  # The pron lengths of all the segments
        for length in range(1, max_len_pron + 1):
            num_available_prons = self.index.get_num_prons(pron_len=length)
            if num_available_prons == 0:
                LOGGER.warning(f"Found no examples for pron_len={length}")
                continue
            desired_samples = max_tot_segments * num_phones_to_prior.get(length, 0)

            # If there aren't many prons available, reduce the number of samples
            # to avoid overtraining on a few prons
            desired_samples = int(min(desired_samples, 3 * num_available_prons))

            LOGGER.info(f"{length=} {desired_samples=}")
            self.pron_lengths += [length] * desired_samples

        random.shuffle(self.pron_lengths)

        LOGGER.info(f"Total samples={len(self.pron_lengths)}")

    def num_segments(self):
        return len(self.pron_lengths)

    def get(self, idx):

        pron_length = self.pron_lengths[idx]

        # Choose a pron index
        pron_idx = self.index.get_random_pron_idx(pron_len=pron_length, exclude_pron_idx=None)

        # Choose a random LSS
        lss = self.index.get_random_lss(pron_idx=pron_idx)

        tokens = self.get_tokens(lss)

        return {
            "long_segment_num": lss.long_seg_num,
            "start_frame": tokens[0]["start_frame"],
            "end_frame": tokens[-1]["end_frame"],
            "orth": " ".join([token["orth"] for token in tokens]),
            "pron": " ".join([token["pron"] for token in tokens]),
        }

    def get_tokens(self, lss: LongSegSpan):
        long_segment = self.long_segments[lss.long_seg_num]
        span_info = self.len_to_span_info[long_segment.length]
        span = span_info.idx_to_span[lss.span_idx]
        tokens = long_segment.token_list[span.start : span.end]
        return tokens


def get_stats(segments):
    char_to_count = {}
    for segment in segments:
        for ch in segment["orth"]:
            if ch not in char_to_count:
                char_to_count[ch] = 0
            char_to_count[ch] += 1
    return {"char_to_count": dict(sorted(char_to_count.items(), key=lambda x: x[1], reverse=True))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-fn", action="store", type=str, required=True)
    parser.add_argument("--index-fn", action="store", type=str, required=True)
    parser.add_argument("--output-pkl", action="store", type=str, required=True)
    parser.add_argument("--output-stats-json", action="store", type=str, required=True)
    parser.add_argument("--pron-len-dist", action="store", type=str, required=True)
    parser.add_argument("--max-segments", action="store", type=int, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    extractor = RandomSegmentExtractor(
        long_segment_metadata_fn=args.metadata_fn,
        index_fn=args.index_fn,
        pron_len_dist=args.pron_len_dist,
        max_tot_segments=args.max_segments,
    )

    segments = []

    for idx in range(extractor.num_segments()):
        segments.append(extractor.get(idx))

    with gzip.open(args.output_pkl, "wb") as fobj:
        pickle.dump(segments, fobj)

    with open(args.output_stats_json, "w", encoding="utf8") as fobj:
        json.dump(get_stats(segments), fobj, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
