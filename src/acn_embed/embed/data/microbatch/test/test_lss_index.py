#!/usr/bin/env python3

import unittest

from acn_embed.embed.data.microbatch.long_segment import LongSegSpan as lss
from acn_embed.embed.data.microbatch.lss_index import LSSIndex


class TestLssIndex(unittest.TestCase):
    def test1(self):
        big_dict1 = {
            1: [[lss(10, 12), lss(11, 4)], [lss(100, 200)]],
            2: [[lss(3, 13), lss(111, 444), lss(222, 333)]],
            3: [
                [lss(3, 13), lss(111, 444), lss(1, 40)],
                [lss(3, 13), lss(111, 444), lss(1, 4)],
                [lss(3, 13), lss(111, 444)],
            ],
            4: [
                [lss(23, 13), lss(111, 444), lss(2, 41)],
                [lss(26, 16)],
                [lss(24, 14), lss(112, 445), lss(2, 42)],
                [lss(27, 18)],
                [lss(25, 15), lss(113, 446)],
            ],
        }
        index = LSSIndex.build_from_pron_len_to_long_seg_spans(big_dict1)

        self.assertEqual(index.get_num_prons(1), 2)
        self.assertEqual(index.get_num_prons(2), 1)

        pron_idx_list = index.list_pron_idx(1, None)

        lss_idx_list = list(index.iter_lss_idx(pron_idx_list[0]))
        self.assertEqual(len(lss_idx_list), 2)
        self.assertEqual(index.get_lss(lss_idx_list[0]), lss(10, 12))
        self.assertEqual(index.get_lss(lss_idx_list[1]), lss(11, 4))

        lss_idx_list = list(index.iter_lss_idx(pron_idx_list[1]))
        self.assertEqual(len(lss_idx_list), 1)
        self.assertEqual(index.get_lss(lss_idx_list[0]), lss(100, 200))

        pron_idx_list = index.list_pron_idx(2, None)

        lss_idx_list = list(index.iter_lss_idx(pron_idx_list[0]))
        self.assertEqual(len(lss_idx_list), 3)
        self.assertEqual(index.get_lss(lss_idx_list[0]), lss(3, 13))
        self.assertEqual(index.get_lss(lss_idx_list[1]), lss(111, 444))
        self.assertEqual(index.get_lss(lss_idx_list[2]), lss(222, 333))

        pron_idx_list = index.list_pron_idx(pron_len=3, exclude_pron_idx=4)
        self.assertListEqual(pron_idx_list, [3, 5])

        self.assertEqual(index.get_num_duplicate_prons(1), 1)
        self.assertEqual(index.get_num_duplicate_prons(2), 1)
        self.assertEqual(index.get_num_duplicate_prons(3), 3)

        pron_idx = index.get_random_pron_idx(pron_len=3, exclude_pron_idx=4)
        self.assertIn(pron_idx, [3, 5])

        lss_list = index.get_random_lss_list(3, 10)
        for _lss in lss_list:
            self.assertIn(_lss, [lss(3, 13), lss(111, 444), lss(1, 40)])

        pron_idx_list = index.list_dup_pron_idx(4)
        self.assertEqual(pron_idx_list[0], 6)
        self.assertEqual(pron_idx_list[1], 8)
        self.assertEqual(pron_idx_list[2], 10)

        for _ in range(5):
            pron_idx = index.get_random_dup_pron_idx(4)
            self.assertIn(pron_idx, [6, 8, 10])


if __name__ == "__main__":
    unittest.main()
