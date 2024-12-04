#!/usr/bin/env python3
import argparse
import sys

from acn_embed.embed.train.g.g_trainer import GTrainer
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


# pylint: disable=duplicate-code


def main():
    parser = argparse.ArgumentParser(
        description="Train G embedder", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    GTrainer.add_args(parser)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    model_spec = {
        "class": "GLstmEmbedder",
        "class_args": {
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

    trainer = GTrainer(args=args, model_spec=model_spec, scheduler_spec=scheduler_spec)

    trainer.run()


if __name__ == "__main__":
    main()
