import pickle

import numpy as np


class CompactNumericIndex:
    def __init__(self, main_idx_2_val_idx, val_arr):
        self.main_idx_2_val_idx = main_idx_2_val_idx
        self.val_arr = val_arr

    def get_num_vals(self, main_idx):
        return self.main_idx_2_val_idx[main_idx + 1] - self.main_idx_2_val_idx[main_idx]

    def iter_val_idx(self, main_idx):
        return range(self.main_idx_2_val_idx[main_idx], self.main_idx_2_val_idx[main_idx + 1])

    def get_val(self, val_idx):
        return self.val_arr[val_idx]

    @staticmethod
    def build_from_dict(dic):
        main_idx_2_val_idx = [0]
        val_list = []
        for main_idx in range(max(dic.keys()) + 1):
            val_list += dic.get(main_idx, [])
            main_idx_2_val_idx.append(len(val_list))
        return CompactNumericIndex(
            main_idx_2_val_idx=np.array(main_idx_2_val_idx).astype(np.int32),
            val_arr=np.array(val_list).astype(np.int32),
        )

    @staticmethod
    def from_fobj(fobj):
        main_idx_2_val_idx = pickle.load(fobj)
        val_arr = pickle.load(fobj)
        return CompactNumericIndex(main_idx_2_val_idx=main_idx_2_val_idx, val_arr=val_arr)

    def to_fobj(self, fobj):
        pickle.dump(self.main_idx_2_val_idx, fobj)
        pickle.dump(self.val_arr, fobj)
