import pickle
import random

import numpy as np


class CompactNumericRangeIndex:
    def __init__(self, main_idx_2_range_idx, range_idx_2_val_idx):
        self.main_idx_2_range_idx = main_idx_2_range_idx
        self.range_idx_2_val_idx = range_idx_2_val_idx

    def get_num_range(self, main_idx):
        return self.main_idx_2_range_idx[main_idx + 1] - self.main_idx_2_range_idx[main_idx]

    def get_num_val(self, range_idx):
        return self.range_idx_2_val_idx[range_idx + 1] - self.range_idx_2_val_idx[range_idx]

    def iter_range_idx(self, main_idx):
        return range(self.main_idx_2_range_idx[main_idx], self.main_idx_2_range_idx[main_idx + 1])

    def list_range_idx(self, main_idx, exclude_range_idx):
        # fmt: off
        if (
            (exclude_range_idx is not None) and
            self.main_idx_2_range_idx[main_idx]
            <= exclude_range_idx
            < self.main_idx_2_range_idx[main_idx + 1]
        ):
            return (
                list(range(self.main_idx_2_range_idx[main_idx], exclude_range_idx)) +
                list(range(exclude_range_idx + 1, self.main_idx_2_range_idx[main_idx + 1]))
            )
        # fmt: on
        return list(self.iter_range_idx(main_idx))

    def iter_val_idx(self, range_idx):
        return range(self.range_idx_2_val_idx[range_idx], self.range_idx_2_val_idx[range_idx + 1])

    def list_val_idx(self, range_idx):
        return list(self.iter_val_idx(range_idx))

    def get_random_range_idx(self, main_idx, exclude_range_idx):
        if exclude_range_idx is None:
            return random.choice(self.iter_range_idx(main_idx))
        return random.choice(self.list_range_idx(main_idx, exclude_range_idx))

    def get_random_val_idx(self, range_idx):
        return random.choice(self.iter_val_idx(range_idx))

    def get_random_val_idx_list(self, range_idx, k):
        val_idx_list = self.list_val_idx(range_idx)
        if len(val_idx_list) <= k:
            return val_idx_list
        return random.sample(val_idx_list, k)

    @staticmethod
    def build_from_dict(dic):
        val_idx = 0
        main_idx_2_range_idx = [0]
        range_idx_2_val_idx = [0]
        for main_idx in range(max(dic.keys()) + 1):
            length_list = dic.get(main_idx, [])
            for length in length_list:
                val_idx += length
                range_idx_2_val_idx.append(val_idx)
            main_idx_2_range_idx.append(len(range_idx_2_val_idx) - 1)
        return CompactNumericRangeIndex(
            main_idx_2_range_idx=np.array(main_idx_2_range_idx).astype(np.int32),
            range_idx_2_val_idx=np.array(range_idx_2_val_idx).astype(np.int32),
        )

    @staticmethod
    def from_fobj(fobj):
        main_idx_2_range_idx = pickle.load(fobj)
        range_idx_2_val_idx = pickle.load(fobj)
        return CompactNumericRangeIndex(
            main_idx_2_range_idx=main_idx_2_range_idx, range_idx_2_val_idx=range_idx_2_val_idx
        )

    def to_fobj(self, fobj):
        pickle.dump(self.main_idx_2_range_idx, fobj)
        pickle.dump(self.range_idx_2_val_idx, fobj)
