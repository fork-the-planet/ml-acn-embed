#!/usr/bin/env python3
import argparse
import gzip
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

from acn_embed.embed.data.microbatch.long_segment import LongSegment
from acn_embed.util.data.storage import H5FeatStoreWriter, H5FeatStoreReader
from acn_embed.util.data.transcription import read_tran_utts
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class LongSegmenter:

    def __init__(self, args):

        self.args = args
        self.frame_rate_ms = args.frame_rate_ms
        os.makedirs(args.output_path, exist_ok=True)
        self.h5_fn = os.path.join(args.output_path, "amfeat.h5")
        self.id_to_long_segment_list = {align_id: [] for align_id in self.args.align_id}
        self.arr_dim = None
        self._feat_writer = None
        self.orth_to_count = defaultdict(int)
        self.pron_to_count = defaultdict(int)
        self._h5_reader = None

    def run(self):

        self._feat_writer = H5FeatStoreWriter(path=self.h5_fn, compress=True)
        self._extract_long_segments()
        self._feat_writer.close()
        print(f"Wrote {self.h5_fn}")

        for align_id in self.args.align_id:
            metadata_fn = os.path.join(self.args.output_path, f"metadata.{align_id}.pkl.gz")
            with gzip.open(metadata_fn, "wb") as fobj:
                pickle.dump(self.id_to_long_segment_list[align_id], fobj)
            print(f"Wrote {metadata_fn}")

        assert self.arr_dim is not None

    def _get_tokens_for_first_align_id(self, utt):

        align_id = self.args.align_id[0]

        if self._utt_has_no_tokens(utt, align_id):
            return []

        tokens = [token for token in utt[align_id][0]["tokens"] if self._token_is_valid(token)]

        # Keep track of how many times the same utt has been seen and apply limits
        _orth = " ".join([token["orth"] for token in tokens])
        _pron = " ".join([token["pron"] for token in tokens])
        if (
            self.orth_to_count[_orth] >= self.args.limit_per_utt
            or self.pron_to_count[_pron] >= self.args.limit_per_utt
        ):
            return []

        self.orth_to_count[_orth] += 1
        self.pron_to_count[_pron] += 1
        return tokens

    def _get_tokens(self, utt, align_id, min_start_ms, max_end_ms):
        if self._utt_has_no_tokens(utt, align_id):
            return []
        tokens = [
            token
            for token in utt[align_id][0]["tokens"]
            if (
                self._token_is_valid(token)
                and token["start_ms"] >= min_start_ms
                and token["end_ms"] <= max_end_ms
            )
        ]
        return tokens

    @staticmethod
    def _utt_has_no_tokens(utt, align_id):
        return (not utt[align_id]) or (not utt[align_id][0]) or (not utt[align_id][0]["tokens"])

    def _token_is_valid(self, token):
        return (
            (token["end_ms"] - token["start_ms"] >= self.args.frame_rate_ms)
            and token["orth"]
            and token["pron"]
        )

    def _extract_metadata(self, utt_info, segment_num):

        align_id_to_tokens = {}

        # Get tokens for the first ("base") align_id
        if not (tokens := self._get_tokens_for_first_align_id(utt_info)):
            return None, None
        align_id_to_tokens[self.args.align_id[0]] = tokens
        min_start_ms = tokens[0]["start_ms"]
        max_end_ms = tokens[-1]["end_ms"]
        long_seg_frame_start = tokens[0]["start_ms"] // self.args.frame_rate_ms
        long_seg_frame_end = tokens[-1]["end_ms"] // self.args.frame_rate_ms

        # Get tokens for the rest of the align_ids
        for align_id in self.args.align_id[1:]:
            if not (tokens := self._get_tokens(utt_info, align_id, min_start_ms, max_end_ms)):
                return None, None
            align_id_to_tokens[align_id] = tokens

        for align_id in self.args.align_id:
            self.id_to_long_segment_list[align_id].append(
                LongSegment(
                    num=segment_num,
                    token_list=[
                        {
                            "orth": token["orth"],
                            "pron": token["pron"],
                            "start_frame": token["start_ms"] // self.args.frame_rate_ms
                            - long_seg_frame_start,
                            "end_frame": token["end_ms"] // self.args.frame_rate_ms
                            - long_seg_frame_start,
                        }
                        for token in align_id_to_tokens[align_id]
                    ],
                    utt_id=utt_info["utt_id"],
                    ms_offset=min_start_ms,
                )
            )

        return long_seg_frame_start, long_seg_frame_end

    def _write_long_segment_feats(
        self, utt_info, segment_num, long_seg_frame_start, long_seg_frame_end
    ):
        h5_fn = os.path.join(self.args.src_h5_base_path, f"amfeat.{utt_info['h5_file_idx']}.h5")

        if self._h5_reader is None or self._h5_reader.path != h5_fn:
            LOGGER.info(f"Reading from {h5_fn}")
            self._h5_reader = H5FeatStoreReader(h5_fn)

        nparr = self._h5_reader.get_nparr(utt_info["h5_num"])[
            long_seg_frame_start:long_seg_frame_end
        ]

        assert np.all(np.isfinite(nparr)), f"Non-finite vals found for {utt_info['utt_id']}"

        if self.arr_dim is None:
            self.arr_dim = nparr.shape[1]
            LOGGER.info(f"Array dimensions = {self.arr_dim}")

        assert self.arr_dim == nparr.shape[1], "Arr dimensions should always be the same"
        self._feat_writer.add_nparr(num=segment_num, nparr=nparr.astype(np.float32))

    def _extract_long_segments(self):

        segment_num = 0

        for _num_seen, utt_info in enumerate(read_tran_utts(self.args.src_tran)):

            if _num_seen % 1000 == 0:
                LOGGER.info(f"utts seen={_num_seen}")

            long_seg_frame_start, long_seg_frame_end = self._extract_metadata(
                utt_info, segment_num
            )

            if (long_seg_frame_start is None) or (long_seg_frame_end is None):
                continue

            self._write_long_segment_feats(
                utt_info, segment_num, long_seg_frame_start, long_seg_frame_end
            )

            segment_num += 1

            if self.args.max_segments and segment_num >= self.args.max_segments:
                LOGGER.info(f"Reached limit. len={segment_num}. Stopping.")
                return

        LOGGER.info(f"Extracted total segs={segment_num}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src-tran", type=str, required=True, help=" ")
    parser.add_argument("--src-h5-base-path", type=str, required=True, help=" ")
    parser.add_argument("--frame-rate-ms", type=int, default=10, help=" ")
    parser.add_argument("--limit-per-utt", type=int, default=100, help=" ")
    parser.add_argument("--max-segments", type=int, default=None, help=" ")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--align-id", type=str, nargs="+", required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    segmenter = LongSegmenter(args)
    segmenter.run()


if __name__ == "__main__":
    main()
