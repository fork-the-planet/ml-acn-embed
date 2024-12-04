#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

from acn_embed.fe_am.nnet.infer.am_inference import AmInference, AmPriorType
from acn_embed.util.data.storage import H5FeatStoreReader, H5FeatStoreWriter
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--fbank-h5", type=Path, required=True)
    parser.add_argument("--am-dir", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    am_inference = AmInference(
        model_dir=args.am_dir,
        boost_sil_log_prior=0,
        prior_type=AmPriorType.NONE,  # We get posteriors
    )

    h5_reader = H5FeatStoreReader(path=args.fbank_h5)
    h5_writer = H5FeatStoreWriter(path=args.output_h5, compress=True)

    LOGGER.info(f"{h5_reader.max_num=}")

    for utt_idx in range(h5_reader.max_num + 1):
        nparr = h5_reader.get_nparr(utt_idx)
        amfeats = am_inference.infer_from_fbank_tensor(
            fbank_tensor=torch.tensor(nparr).to(torch.float32), return_as_nparr=True
        )
        h5_writer.add_nparr(utt_idx, amfeats)

    h5_reader.close()
    h5_writer.close()


if __name__ == "__main__":
    main()
