#!/usr/bin/env python3

import argparse
import sys

import numpy as np

from acn_embed.embed.embedder.text_embedder import TextEmbedder
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def get_confusion_mat(embedder: TextEmbedder, sigma):
    subwords = sorted(embedder.model.subword_to_idx.keys())
    num_subwords = len(subwords)
    mat = np.zeros((num_subwords, num_subwords), dtype=np.float32)
    for i1, subword1 in enumerate(subwords):
        for i2, subword2 in enumerate(subwords):
            if i1 == i2:
                continue
            if sigma == 0:
                mat[i1, i2] = 1
            else:
                mat[i1, i2] = 1 - np.exp(
                    -np.sum(
                        (
                            embedder.get_embedding([[subword1]]).detach().cpu().numpy()
                            - embedder.get_embedding([[subword2]]).detach().cpu().numpy()
                        )
                        ** 2
                    )
                    / (8 * sigma * sigma)
                )
    return mat


def _get_backtrace(backtrace):
    codes = []
    x, y = backtrace.shape
    x -= 1
    y -= 1
    while x > 0 or y > 0:
        _bt = backtrace[x, y]
        codes.append(_bt)
        if _bt in [1, 4]:
            assert x > 0 and y > 0
            x -= 1
            y -= 1
        elif _bt == 2:
            assert x > 0
            x -= 1
        elif _bt == 3:
            assert y > 0
            y -= 1
        else:
            raise RuntimeError("coding error")
    assert backtrace[0, 0] == 4
    codes.reverse()
    return codes


def compute_min_ed_distance(seq1, seq2, subst_cost_mat, insert_cost, delete_cost):
    """Simple (unoptimized) min edit distance"""
    costmat = np.zeros((len(seq1) + 1, len(seq2) + 1), np.float32)
    backtrace = np.zeros(
        (len(seq1) + 1, len(seq2) + 1), np.int32
    )  # 1: subst, 2: ins, 3: del, 4:orig
    backtrace[0, 0] = 4
    for x in range(len(seq1) + 1):
        for y in range(len(seq2) + 1):
            if x == 0 and y == 0:
                continue
            subst_cost = np.inf
            ins_cost = np.inf
            del_cost = np.inf
            if x > 0 and y > 0 and backtrace[x - 1, y - 1] != 0:
                subst_cost = costmat[x - 1, y - 1] + subst_cost_mat[seq1[x - 1], seq2[y - 1]]
            if x > 0 and backtrace[x - 1, y] != 0:
                ins_cost = costmat[x - 1, y] + insert_cost
            if y > 0 and backtrace[x, y - 1] != 0:
                del_cost = costmat[x, y - 1] + delete_cost
            costs = [subst_cost, ins_cost, del_cost]
            k = np.argmin(costs)
            if not np.isfinite(costs[k]):
                continue
            costmat[x, y] = costs[k]
            backtrace[x, y] = k + 1
            if k == 0 and seq1[x - 1] == seq2[y - 1]:
                backtrace[x, y] = 4

    LOGGER.debug(_get_backtrace(backtrace))
    return costmat[-1, -1]


class PronMinEditDistance:
    def __init__(self, model_dir, sigma, insert_cost, delete_cost):
        self.embedder = TextEmbedder(model_dir=model_dir, text_type="phone")
        self.confusion_mat = get_confusion_mat(
            self.embedder, sigma if (sigma is not None) else self.embedder.sigma
        )
        LOGGER.info(self.confusion_mat)
        self.insert_cost = insert_cost
        self.delete_cost = delete_cost

    def get_distance(self, phones1, phones2):
        return compute_min_ed_distance(
            seq1=[self.embedder.model.subword_to_idx[ph] for ph in phones1],
            seq2=[self.embedder.model.subword_to_idx[ph] for ph in phones2],
            subst_cost_mat=self.confusion_mat,
            insert_cost=self.insert_cost,
            delete_cost=self.delete_cost,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("model_dir", action="store", type=str)
    parser.add_argument("phones1", action="store", type=str)
    parser.add_argument("phones2", action="store", type=str)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    ed = PronMinEditDistance(args.model_dir, 1, 1, 1)
    print(ed.get_distance(args.phones1.split(), args.phones2.split()))
