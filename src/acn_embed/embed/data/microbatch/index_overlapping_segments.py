#!/usr/bin/env python3

import argparse
import gzip
import pickle
import random
import sys

from acn_embed.embed.data.microbatch.long_segment import LongSegSpan
from acn_embed.embed.data.microbatch.lss_index import LSSIndex
from acn_embed.embed.data.microbatch.span_info import SpanInfo
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class OverlapIndexer:
    def __init__(self, metadata_fn, max_audio_len_frame, max_pron_len):
        with gzip.open(metadata_fn, "rb") as fobj:
            self.long_segments = pickle.load(fobj)

        self.len_to_span_info = {}
        for long_segment in self.long_segments:
            long_segment.add_span_indexing(max_audio_len_frame, max_pron_len)
            if (long_segment.length) not in self.len_to_span_info:
                self.len_to_span_info[long_segment.length] = SpanInfo(long_segment.length)

        self.segments = []

        # Map from pron_len (int) -> list of list of long_seg_span
        # Each sub list is for the same pron
        self.pron_len_to_long_seg_spans = {}

    def index_long_seg_spans(self):
        pron_to_long_seg_spans = {}
        for long_segment in self.long_segments:
            for span_idx, pron in long_segment.span_idx_to_pron.items():
                if pron is None:
                    continue
                if pron not in pron_to_long_seg_spans:
                    pron_to_long_seg_spans[pron] = []
                pron_to_long_seg_spans[pron].append(LongSegSpan(long_segment.num, span_idx))
        for pron, long_seg_spans in pron_to_long_seg_spans.items():
            pron_len = len(pron.split())
            if pron_len not in self.pron_len_to_long_seg_spans:
                self.pron_len_to_long_seg_spans[pron_len] = []
            self.pron_len_to_long_seg_spans[pron_len].append(long_seg_spans)

    def sanity_check_and_show_examples(self):

        for length in sorted(self.pron_len_to_long_seg_spans.keys()):
            LOGGER.info(f"-- Examples with length {length} --")
            lss_lists = self.pron_len_to_long_seg_spans[length]
            lss_list = random.choice(lss_lists)
            lss_list_ = random.sample(lss_list, min(3, len(lss_list)))
            pron = None
            for lss in lss_list_:
                longseg = self.long_segments[lss.long_seg_num]
                assert longseg.num == lss.long_seg_num
                _pron = longseg.span_idx_to_pron[lss.span_idx]
                if _pron is None:
                    continue
                if pron is None:
                    pron = _pron
                assert longseg.span_idx_to_pron[lss.span_idx] == pron
                span_info = self.len_to_span_info[longseg.length]
                span = span_info.idx_to_span[lss.span_idx]
                tokens = longseg.token_list[span.start : span.end]
                LOGGER.info(
                    [
                        (longseg.num, span.start, span.end, token["orth"], token["pron"])
                        for token in tokens
                    ]
                )

        # pylint: disable=consider-using-generator
        for length in sorted(self.pron_len_to_long_seg_spans.keys()):
            lss_lists = self.pron_len_to_long_seg_spans[length]
            num_lss = sum([len(lss_list) for lss_list in lss_lists])
            num_dup_prons = sum([1 for lss_list in lss_lists if len(lss_list) > 1])
            LOGGER.info(f"{length=} {num_lss=:,} {num_dup_prons=:,}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--metadata-fn", type=str, required=True, help=" ")
    parser.add_argument("--output-fn", type=str, required=True, help=" ")
    parser.add_argument("--max-audio-len-frame", type=int, required=True, help=" ")
    parser.add_argument("--max-pron-len", type=int, required=True, help=" ")
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    segmenter = OverlapIndexer(
        metadata_fn=args.metadata_fn,
        max_audio_len_frame=args.max_audio_len_frame,
        max_pron_len=args.max_pron_len,
    )
    segmenter.index_long_seg_spans()
    segmenter.sanity_check_and_show_examples()

    lss_index = LSSIndex.build_from_pron_len_to_long_seg_spans(
        segmenter.pron_len_to_long_seg_spans
    )

    with gzip.open(args.output_fn, "wb") as fobj:
        lss_index.to_fobj(fobj)


if __name__ == "__main__":
    main()
