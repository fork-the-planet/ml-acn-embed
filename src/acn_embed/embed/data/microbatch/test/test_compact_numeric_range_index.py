#!/usr/bin/env python3

import unittest

from acn_embed.embed.data.microbatch.compact_numeric_range_index import CompactNumericRangeIndex


class TestCompactNumericRangeIndex(unittest.TestCase):
    def test1(self):
        dict1 = {1: [2, 1], 2: [3], 3: [3, 3, 2]}

        # This is derived from
        # dict1 = {
        #     1: [
        #         [0, 1],
        #         [2]
        #     ],
        #     2: [
        #         [3, 4, 5]
        #     ],
        #     3: [
        #         [6, 7, 8],
        #         [9, 10, 11],
        #         [12, 13]
        #     ]
        # }

        index = CompactNumericRangeIndex.build_from_dict(dict1)
        print(index.main_idx_2_range_idx, index.range_idx_2_val_idx)

        self.assertEqual(index.get_num_range(0), 0)
        self.assertEqual(index.get_num_range(1), 2)
        self.assertEqual(index.get_num_range(2), 1)
        self.assertEqual(index.get_num_range(3), 3)

        range_idx_list = index.list_range_idx(1, None)
        self.assertEqual(list(index.list_val_idx(range_idx_list[0])), [0, 1])
        self.assertEqual(list(index.list_val_idx(range_idx_list[1])), [2])

        range_idx_list = index.list_range_idx(2, None)
        self.assertEqual(list(index.list_val_idx(range_idx_list[0])), [3, 4, 5])

        range_idx_list = index.list_range_idx(3, None)
        print(range_idx_list)
        self.assertEqual(list(index.list_val_idx(range_idx_list[0])), [6, 7, 8])
        self.assertEqual(list(index.list_val_idx(range_idx_list[1])), [9, 10, 11])
        self.assertEqual(list(index.list_val_idx(range_idx_list[2])), [12, 13])

        range_idx_list = index.list_range_idx(3, exclude_range_idx=4)
        print(range_idx_list)
        self.assertEqual(list(index.list_val_idx(range_idx_list[0])), [6, 7, 8])
        self.assertEqual(list(index.list_val_idx(range_idx_list[1])), [12, 13])


if __name__ == "__main__":
    unittest.main()
