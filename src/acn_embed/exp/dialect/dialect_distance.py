#!/usr/bin/env python3
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def get_dist_mat(pickle_file, sigma):
    with open(pickle_file, "rb") as fobj:
        pickle.load(fobj)
        dr_to_embeddings = pickle.load(fobj)

    centroid = {}
    for dr in range(1, 9):
        # List, where each elem is a speaker's embeddings for every word shape (11, d)
        # where 11 is the number of words and d is the embedding dimensions
        embeddings = dr_to_embeddings[f"dr{dr}"]
        emb = np.concatenate(
            [np.expand_dims(emb.detach().cpu().numpy(), 0) for emb in embeddings], axis=0
        )
        centroid[dr] = np.mean(emb, axis=0)  # shape (11, d)

    dist_mat = np.zeros((10, 10))
    for row in range(2, 9):
        for col in range(1, row):
            dist = np.mean(
                1
                - np.exp(
                    -np.sum(np.power(centroid[row] - centroid[col], 2.0), axis=-1)
                    / (8 * sigma * sigma)
                )
            )
            dist_mat[row, col] = dist
            dist_mat[col, row] = dist
    return dist_mat


def main():
    parser = argparse.ArgumentParser(
        description="Compute dialect dissimilarities",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pickle_file", type=Path, default=None)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--addtree", action="store_true", default=False)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    dr_info = {
        1: "New England",
        2: "Northern",
        3: "North Midland",
        4: "South Midland",
        5: "Southern",
        7: "Western",
    }

    dist_mat = get_dist_mat(args.pickle_file, args.sigma)

    if args.addtree:
        # The ordering here only has a cosmetic effect. The actual result does not change.
        show_dr = [1, 4, 5, 2, 3, 7]
        print("nonumber,residuals,minvarroot")
        print("dialects")
        print("6 DISSIMILARITIES LOWERHALF 15 10")
        for row in range(1, 6):
            for col in range(0, row):
                print(
                    f"{dist_mat[show_dr[row], show_dr[col]]:.10f}",
                    end=" " if col < row - 1 else "\n",
                )
    else:
        show_dr = [1, 2, 3, 4, 5, 7]
        for row in range(1, 6):
            print(f"dr{show_dr[row]} & ", end="")
            for col in range(0, row):
                print(
                    f"{dist_mat[show_dr[row], show_dr[col]]:.5f}",
                    end=" & " if col < row - 1 else "",
                )
            for col in range(row, 5):
                print(" & *      ", end="")
            print(" \\\\")

    for dr in show_dr:
        print(dr_info[dr])


if __name__ == "__main__":
    main()
