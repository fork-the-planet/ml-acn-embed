#!/usr/bin/env python3

import unittest

import numpy as np

from acn_embed.exp.oovrecovery.min_ed import compute_min_ed_distance


def uniform_cost_mat(dim):
    return np.ones((dim, dim)) - np.eye(dim)


class TestMinEditDistance(unittest.TestCase):
    def test_same(self):
        self.assertAlmostEqual(
            compute_min_ed_distance(
                seq1=[0, 1, 2, 3, 4, 5],
                seq2=[0, 1, 2, 3, 4, 5],
                subst_cost_mat=uniform_cost_mat(6),
                insert_cost=1,
                delete_cost=1,
            ),
            0,
        )

    def test_1sub(self):
        self.assertAlmostEqual(
            compute_min_ed_distance(
                seq1=[0, 1, 2, 3, 4, 5],
                seq2=[0, 1, 2, 0, 4, 5],
                subst_cost_mat=uniform_cost_mat(6),
                insert_cost=1,
                delete_cost=1,
            ),
            1,
        )
        self.assertAlmostEqual(
            compute_min_ed_distance(
                seq1=[0, 1, 2, 3, 4, 5],
                seq2=[0, 1, 2, 0, 4, 5],
                subst_cost_mat=0.8 * uniform_cost_mat(6),
                insert_cost=1,
                delete_cost=1,
            ),
            0.8,
        )

    def test_sid(self):
        # ins, del, sub
        self.assertAlmostEqual(
            compute_min_ed_distance(
                seq1=[0, 1, 2, 2, 3, 4, 6, 5, 8],
                seq2=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                subst_cost_mat=0.5 * uniform_cost_mat(9),
                insert_cost=0.4,
                delete_cost=0.6,
            ),
            0.4 + 0.6 + 0.5,
        )

        # ins, del, ins, del
        self.assertAlmostEqual(
            compute_min_ed_distance(
                seq1=[0, 1, 2, 0, 3, 5, 6, 0, 7],
                seq2=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                subst_cost_mat=0.1 * uniform_cost_mat(9),
                insert_cost=0.01,
                delete_cost=0.001,
            ),
            0.01 + 0.001 + 0.01 + 0.001,
            places=4,
        )

        # del, del, ins, del, sub, ins, ins, ins
        self.assertAlmostEqual(
            compute_min_ed_distance(
                seq1=[2, 3, 0, 4, 5, 7, 8, 0, 10, 11, 0, 12, 13, 14, 15],
                seq2=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                subst_cost_mat=5.1 * uniform_cost_mat(16),
                insert_cost=3.9,
                delete_cost=6.2,
            ),
            6.2 + 6.2 + 3.9 + 6.2 + 5.1 + 3.9 + 3.9 + 3.9,
            places=4,
        )

        # ins, ins, sub, del, del, ins, del, del
        self.assertAlmostEqual(
            compute_min_ed_distance(
                seq1=[2, 2, 0, 1, 2, 0, 4, 5, 7, 9, 0, 10, 11],
                seq2=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                subst_cost_mat=5.1 * uniform_cost_mat(14),
                insert_cost=3.9,
                delete_cost=6.2,
            ),
            3.9 + 3.9 + 5.1 + 6.2 + 6.2 + 3.9 + 6.2 + 6.2,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
