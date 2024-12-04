#!/usr/bin/env python3

import unittest

from acn_embed.embed.data.microbatch.compact_numeric_index import CompactNumericIndex


class TestCompactNumericIndex(unittest.TestCase):
    def test1(self):
        dic = {1: [10, 20], 2: [30], 3: [40, 50, 60], 5: [70, 80]}
        index = CompactNumericIndex.build_from_dict(dic)

        print(index.main_idx_2_val_idx, index.val_arr)

        self.assertEqual(index.get_num_vals(0), 0)
        self.assertEqual(index.get_num_vals(1), 2)
        self.assertEqual(index.get_num_vals(2), 1)
        self.assertEqual(index.get_num_vals(3), 3)
        self.assertEqual(index.get_num_vals(4), 0)
        self.assertEqual(index.get_num_vals(5), 2)

        val_idx_list = list(index.iter_val_idx(3))
        self.assertEqual(index.get_val(val_idx_list[0]), 40)
        self.assertEqual(index.get_val(val_idx_list[1]), 50)
        self.assertEqual(index.get_val(val_idx_list[2]), 60)


if __name__ == "__main__":
    unittest.main()
