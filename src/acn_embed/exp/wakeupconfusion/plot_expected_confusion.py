#!/usr/bin/env python3
import argparse
import bisect
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def plot(wakeup_words, confs, lm_scales, output):
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 16,
        }
    )
    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
    for conf, marker in zip(confs, ["o", "^", "s", "*", "D"]):
        plt.plot(lm_scales, conf, marker + ":")
    plt.grid()
    plt.legend(
        wakeup_words,
        prop={"size": 11},
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        labelspacing=0.1,
        ncol=3,
    )
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\log E$")
    plt.gcf().set_figheight(4)
    plt.gcf().set_figwidth(6)
    plt.gcf().tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(output, bbox_inches="tight", pad_inches=0)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Plot expected confusion curves",
    )
    parser.add_argument("--data", action="store", type=Path, required=True)
    parser.add_argument("--min-lm-scale", action="store", type=float, required=True)
    parser.add_argument("--max-lm-scale", action="store", type=float, required=True)
    parser.add_argument("--output", action="store", type=str, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    with args.data.open("rb") as fobj:
        obj = torch.load(fobj, weights_only=True, map_location="cpu")
        (wakeup_words, confs, lm_scales) = (
            obj["wakeup_words"],
            np.array(obj["confs"]),
            obj["lm_scales"],
        )

    wakeup_words = [word.upper().replace("TEE ONE THOUSAND", "T1000") for word in wakeup_words]
    first_idx = bisect.bisect_right(lm_scales, args.min_lm_scale)
    end_idx = bisect.bisect_right(lm_scales, args.max_lm_scale)
    LOGGER.info(f"{first_idx=} {end_idx=}")

    plot(wakeup_words, confs[:, first_idx:end_idx], lm_scales[first_idx:end_idx], args.output)


if __name__ == "__main__":
    main()
