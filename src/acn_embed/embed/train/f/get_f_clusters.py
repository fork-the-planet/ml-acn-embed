#!/usr/bin/env python3

import argparse
import gzip
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from acn_embed.embed.embedder.audio_embedder import AudioEmbedder
from acn_embed.util.base_trainer import base_functions
from acn_embed.util.data.storage import H5FeatStoreReader
from acn_embed.util.logger import get_logger
from acn_embed.util.utils import get_start_end_idx

LOGGER = get_logger(__name__)


class TokenToFoutput:
    def __init__(self, data_path: Path, f_model_dir: Path, orth_or_pron: str, min_count: int):
        self.orth_or_pron = orth_or_pron
        with gzip.open(data_path / "metadata.pkl.gz", "rb") as fobj:
            self.long_segments = pickle.load(fobj)
        self._get_highcount_tokens(min_count=min_count)
        self.feat_store = H5FeatStoreReader(data_path / "amfeat.h5")
        self.embedder = AudioEmbedder(f_model_dir)

    def _get_highcount_tokens(self, min_count: int):
        toklab_counter = Counter()
        for long_segment in self.long_segments:
            for token in long_segment.token_list:
                toklab_counter[token[self.orth_or_pron]] += 1
        self.filtered_counter = Counter()
        for label, count in toklab_counter.items():
            if count >= min_count:
                self.filtered_counter[label] = count
        LOGGER.info(f"{len(toklab_counter)=}")
        LOGGER.info(f"{len(self.filtered_counter)=}")
        self.highcount_toklabs = sorted(self.filtered_counter.keys())

    def run(self):
        toklab_to_f_list = {}
        start_idx, end_idx = get_start_end_idx(
            len(self.long_segments), base_functions.world_size(), base_functions.get_rank() + 1
        )
        LOGGER.info(f"Gathering f-embeddings {start_idx=} {end_idx=}")
        for idx in range(start_idx, end_idx):
            if (idx - start_idx) % 100000 == 0:
                LOGGER.info(f"{idx=}")
            long_segment = self.long_segments[idx]
            assert idx == long_segment.num
            nparr_long_segment = self.feat_store.get_nparr(long_segment.num)
            foutput_arr = self.embedder.get_embedding(
                amfeat_tensor=torch.tensor(nparr_long_segment).to(device=self.embedder.device),
                segments=np.array(
                    [
                        [token["start_frame"] * 10, token["end_frame"] * 10]
                        for token in long_segment.token_list
                    ]
                ),
            )
            if foutput_arr is None:
                continue
            foutput_arr = foutput_arr.detach().cpu().numpy()
            for token, foutput in zip(long_segment.token_list, foutput_arr, strict=True):
                toklab = token[self.orth_or_pron]
                if toklab not in self.highcount_toklabs:
                    continue
                if toklab not in toklab_to_f_list:
                    toklab_to_f_list[toklab] = []
                toklab_to_f_list[toklab].append(foutput)
        return toklab_to_f_list


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data-path", action="store", type=Path, required=True)
    parser.add_argument("--embedder", action="store", type=Path, required=True)
    parser.add_argument("--min-count", action="store", type=int, required=True)
    parser.add_argument("--subword", action="store", choices=["phone", "grapheme"], required=True)
    parser.add_argument("--output-dir", action="store", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    base_functions.dist_init()

    obj = TokenToFoutput(
        data_path=args.data_path,
        f_model_dir=args.embedder,
        orth_or_pron=("pron" if args.subword == "phone" else "orth"),
        min_count=args.min_count,
    )
    toklab_to_f_list = obj.run()
    with open(args.output_dir / f"f-data.{base_functions.get_rank()}.pkl", "wb") as fobj:
        pickle.dump(toklab_to_f_list, fobj)

    base_functions.dist_barrier(timeout_seconds=3600)
    base_functions.dist_shutdown()


if __name__ == "__main__":
    main()
