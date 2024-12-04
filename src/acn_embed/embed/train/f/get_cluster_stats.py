#!/usr/bin/env python3

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from IsoScore import IsoScore
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def read_clusters(data_path_list: list, min_num_samples: int):
    token_to_f = {}
    for data_path in data_path_list:
        with data_path.open("rb") as fobj:
            for token, flist in pickle.load(fobj).items():
                if token not in token_to_f:
                    token_to_f[token] = []
                token_to_f[token] += flist
    token_to_farr = {}
    embed_dim = None
    for token, foutputs in token_to_f.items():
        num_samples = len(foutputs)
        if num_samples < min_num_samples:
            continue
        farr = np.concatenate([np.expand_dims(x, 0) for x in foutputs], axis=0)
        if embed_dim is None:
            embed_dim = farr.shape[1]
        assert embed_dim == farr.shape[1]  # Sanity check
        token_to_farr[token] = farr
    return token_to_farr, embed_dim


def get_cluster_stats(data_path_list: list, min_num_samples: int):
    token_to_farr, embed_dim = read_clusters(
        data_path_list=data_path_list, min_num_samples=min_num_samples
    )
    if token_to_farr is None:
        return None
    num_embeddings = 0
    clusterwise_stds = []
    elementwise_abs_f_sum = 0.0
    isoscores = []
    for _, farr in token_to_farr.items():
        isoscores.append(IsoScore.IsoScore(farr).item())
        elementwise_abs_f_sum += np.sum(np.absolute(farr))
        clusterwise_stds += np.std(farr, axis=0).tolist()
        num_embeddings += farr.shape[0]

    std_of_clusterwise_std = np.std(clusterwise_stds)
    mean_of_elementwise_abs_f = elementwise_abs_f_sum / (num_embeddings * embed_dim)
    mean_of_clusterwise_std = np.mean(clusterwise_stds)

    result = {
        "num_clusters": len(isoscores),
        "mean_isoscore": float(np.mean(isoscores)),
        "std_of_clusterwise_std": float(std_of_clusterwise_std),
        "mean_of_elementwise_abs_f": float(mean_of_elementwise_abs_f),
        "mean_of_clusterwise_std": float(mean_of_clusterwise_std),
        "ratio1": float(std_of_clusterwise_std / mean_of_elementwise_abs_f),
        "ratio2": float(std_of_clusterwise_std / mean_of_clusterwise_std),
    }
    LOGGER.info(json.dumps(result, indent=4))
    return result


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", action="store", type=Path, nargs="+", required=True)
    parser.add_argument("--min-samples", action="store", type=int, required=True)
    parser.add_argument("--output", action="store", type=Path, default=None)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    results = get_cluster_stats(data_path_list=args.data, min_num_samples=args.min_samples)
    if args.output:
        with args.output.open("w", encoding="utf8") as fobj:
            json.dump(results, fobj, indent=4)


if __name__ == "__main__":
    main()
