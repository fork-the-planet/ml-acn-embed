#!/usr/bin/env python3
import argparse
import os
import sys

from acn_embed.embed.train.f.f_trainer import FTrainer
from acn_embed.util.data.storage import get_feat_dim
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train F embedder", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    FTrainer.add_args(parser)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    model_spec = {
        "class": "FLstmEmbedder",
        "class_args": {
            "dim_in": get_feat_dim(os.path.join(args.data_path_cv, "amfeat.h5")),
            "dim_out": args.dim_out,
            "dim_state": args.dim_state,
            "dropout_prob": args.dropout_prob,
            "num_layers": args.num_layers,
        },
    }
    scheduler_spec = {
        "peak_lr": args.max_lrate,
        "warmup_steps_k": args.warmup_steps_k,
        "decay1_steps_k": args.decay1_steps_k,
        "decay2_factor_k": args.decay2_factor_k,
    }

    trainer = FTrainer(
        args=args,
        model_spec=model_spec,
        scheduler_spec=scheduler_spec,
    )

    trainer.run()


if __name__ == "__main__":
    main()
