#!/usr/bin/env python3

import argparse
import sys

import acn_embed.fe_am.nnet.train.get_model_params
from acn_embed.fe_am.nnet.train.fe_am_trainer import FeAmTrainer
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train DNN-HMM acoustic model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    FeAmTrainer.add_args(parser)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    if args.dim_out_fn:
        with open(args.dim_out_fn, "r", encoding="utf8") as fobj:
            args.dim_out = int(fobj.read().strip())

    model_spec = {
        "class": "ConformerAm",
        "class_args": {
            "frame_ms": 10,
            "dim_in": args.acoustic_feat_dim,
            "dim_out": args.dim_out,
            "num_subsample_layers": 0,
        },
    }
    model_spec["class_args"].update(
        acn_embed.fe_am.nnet.train.get_model_params.get_conformer(size=args.model_size)
    )

    scheduler_spec = {
        "peak_lr": args.max_lrate_mult * (model_spec["class_args"]["embedder_dim"] ** (-0.5)),
        "warmup_steps_k": args.warmup_steps_k,
    }

    trainer = FeAmTrainer(args, model_spec=model_spec, scheduler_spec=scheduler_spec)

    trainer.run()


if __name__ == "__main__":
    main()
