#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import kaldi_io
import torch

from acn_embed.fe_am.nnet.infer.am_inference import AmInference, AmPriorType
from acn_embed.fe_am.nnet.infer.inference_dataset import InferenceDataset
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def infer_to_stdout(data_generator: torch.utils.data.DataLoader, am_inference: AmInference):
    num = 0

    with torch.no_grad():

        for collated_data in data_generator:
            padded_input_t, utt_ids, input_len_t = collated_data

            output_t, output_len_t = am_inference.infer_from_fbank_batch_tensor(
                fbank_tensor=padded_input_t, input_len_tensor=input_len_t
            )

            if torch.isnan(output_t).any():
                raise RuntimeError("NaN found in output. Aborting.")

            for _output_t, _output_len_t, utt_id in zip(output_t, output_len_t, utt_ids):
                if num % 1000 == 0:
                    LOGGER.info(f"num={num} utt_id={utt_id}")
                num += 1
                kaldi_io.write_mat(
                    sys.stdout.buffer,
                    _output_t[:_output_len_t, :].detach().cpu().numpy(),
                    key=utt_id,
                )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Runs AM inference and streams binary data to stdout " '(like wspecifier "ark:-")'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--h5", action="store", type=Path, default=None)
    parser.add_argument(
        "--tran", action="store", type=Path, default=None, help="Just to get the utt IDs"
    )
    parser.add_argument("--model-dir", action="store", type=Path, required=True)
    parser.add_argument("--sort-utt-ids", action="store_true", default=False)
    parser.add_argument("--boost-sil-log-prior", action="store", type=float, default=0)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    if args.cuda_device >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device % torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")

    am_inference = AmInference(
        model_dir=args.model_dir,
        boost_sil_log_prior=args.boost_sil_log_prior,
        prior_type=AmPriorType.MARGIN,
        device=device,
    )

    dataset = InferenceDataset(
        h5_fn=args.h5,
        tran_fn=args.tran,
        sort_utt_ids=args.sort_utt_ids,
        min_input_len=am_inference.model.min_input_len,
    )

    data_generator = torch.utils.data.DataLoader(
        batch_size=1,
        collate_fn=dataset.collate,
        dataset=dataset,
        drop_last=False,
        multiprocessing_context="spawn",
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,
        shuffle=False,
    )

    infer_to_stdout(data_generator, am_inference)


if __name__ == "__main__":
    main()
