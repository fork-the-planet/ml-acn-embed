#!/usr/bin/env python3
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from acn_embed.embed.model.get_model import load
from acn_embed.embed.train.g.segment_amfeat_dataset import SegmentAmFeatDataset
from acn_embed.util.base_inference_input import BaseInferenceInput
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class FOutputter:
    def __init__(self, args):
        self.args = args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.fmodel = load(self.args.nnet, self.device)
        self.dataset = SegmentAmFeatDataset(
            h5_path=args.h5, segment_path=args.segments, min_input_len=self.fmodel.min_input_len
        )

    def run(self):
        data_generator = torch.utils.data.DataLoader(
            batch_size=self.args.batch_size,
            collate_fn=self.dataset.collate,
            dataset=self.dataset,
            num_workers=self.args.num_data_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            sampler=None,
            shuffle=False,
        )
        LOGGER.info("Num batches=%d", len(data_generator))
        start_time = time.time()
        all_idx_list = []
        foutput_list = []
        with torch.no_grad():
            for idx_list, padded_input_t, num_sequences, num_frames_t in data_generator:
                all_idx_list += idx_list
                foutput = self.fmodel.forward(
                    BaseInferenceInput(
                        input_t=padded_input_t.to(self.device), input_len_t=num_frames_t
                    )
                )
                foutput_list.append(foutput.cpu().detach().numpy())
                assert foutput.shape == (num_sequences, self.fmodel.dim_out)
                assert len(idx_list) == num_sequences
        LOGGER.info(f"Seen samples: {len(all_idx_list)}")
        elapsed = time.time() - start_time
        LOGGER.info(
            "Elapsed(ms) = %.2f time per sequence(ms) = %.2f",
            1000 * elapsed,
            1000 * elapsed / len(all_idx_list),
        )
        assert sorted(all_idx_list) == all_idx_list
        all_foutput = np.concatenate(foutput_list, axis=0)
        assert len(all_idx_list) == all_foutput.shape[0]
        with open(self.args.output_pkl, "wb") as fobj:
            pickle.dump(all_foutput.shape, fobj)
            pickle.dump(all_foutput, fobj)
        LOGGER.info("Wrote %s", self.args.output_pkl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", action="store", type=int, default=200)
    parser.add_argument("--h5", action="store", type=Path, default=200)
    parser.add_argument("--nnet", action="store", type=Path, required=True)
    parser.add_argument("--num-data-workers", action="store", type=int, default=4)
    parser.add_argument("--output-pkl", action="store", type=Path, required=True)
    parser.add_argument("--segments", action="store", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    foutputter = FOutputter(args)
    foutputter.run()


if __name__ == "__main__":
    main()
