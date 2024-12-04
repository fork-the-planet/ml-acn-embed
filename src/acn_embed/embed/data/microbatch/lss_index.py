import pickle
import random

import numpy as np

from acn_embed.embed.data.microbatch.compact_numeric_index import CompactNumericIndex
from acn_embed.embed.data.microbatch.compact_numeric_range_index import CompactNumericRangeIndex
from acn_embed.embed.data.microbatch.long_segment import LongSegSpan


class LSSIndex:
    """
    Storage of pron_len_to_long_seg_span_lists
    A cnr index is used to map pron_len to pron to lss index
    A cn index is used to map pron_len to pron_idx (for prons that have more than 1 example)
    """

    def __init__(
        self,
        cnr_index: CompactNumericRangeIndex,
        cn_index: CompactNumericIndex,
        ls_num_arr,
        ls_span_idx_arr,
    ):
        self.cnr_index = cnr_index
        self.cn_index = cn_index
        self.ls_num_arr = ls_num_arr
        self.ls_span_idx_arr = ls_span_idx_arr

        # The maximum publicly viewable pron len.
        # Internally, we have an extra "placeholder" pron len
        self.max_pron_len = cnr_index.main_idx_2_range_idx.shape[0] - 2

        # The total number of publicly viewable prons
        # Internally we have an extra "placeholder" pron
        self.num_prons = cnr_index.range_idx_2_val_idx.shape[0] - 1

    def get_num_prons(self, pron_len):
        return self.cnr_index.get_num_range(pron_len)

    def get_num_duplicate_prons(self, pron_len):
        """
        For a given pron_len, returns the number of prons that occur more than once
        """
        return self.cn_index.get_num_vals(pron_len)

    def get_random_pron_idx(self, pron_len, exclude_pron_idx):
        return self.cnr_index.get_random_range_idx(pron_len, exclude_pron_idx)

    def get_random_lss_list(self, pron_idx, k):
        lss_idx_list = self.cnr_index.list_val_idx(pron_idx)
        if len(lss_idx_list) <= k:
            sample_list = lss_idx_list
        else:
            sample_list = random.sample(lss_idx_list, k)
        return [self.get_lss(lss_idx) for lss_idx in sample_list]

    def get_num_samples(self, pron_idx):
        """
        Returns the number of samples in the database for a given pron
        """
        return len(self.cnr_index.list_val_idx(pron_idx))

    def get_lss_list(self, pron_idx):
        """
        Returns a list of all LSS's in the database for a given pron
        """
        return [self.get_lss(lss_idx) for lss_idx in self.cnr_index.list_val_idx(pron_idx)]

    def get_lss(self, lss_idx):
        return LongSegSpan(
            long_seg_num=self.ls_num_arr[lss_idx], span_idx=self.ls_span_idx_arr[lss_idx]
        )

    def get_random_lss(self, pron_idx):
        return self.get_lss(random.choice(self.cnr_index.iter_val_idx(pron_idx)))

    def get_num_lss(self, pron_idx):
        return self.cnr_index.get_num_val(pron_idx)

    def iter_pron_idx(self, pron_len):
        return self.cnr_index.iter_range_idx(pron_len)

    def list_pron_idx(self, pron_len, exclude_pron_idx):
        return self.cnr_index.list_range_idx(pron_len, exclude_pron_idx)

    def iter_lss_idx(self, pron_idx):
        return self.cnr_index.iter_val_idx(pron_idx)

    def list_dup_pron_idx(self, pron_len):
        """
        Iterator over pron_idx for prons that have at least two examples
        """
        return [self.cn_index.get_val(val_idx) for val_idx in self.cn_index.iter_val_idx(pron_len)]

    def get_random_dup_pron_idx(self, pron_len):
        """
        Returns a random pron idx for which at least two examples exist
        """
        idx = random.choice(self.cn_index.iter_val_idx(pron_len))
        return self.cn_index.get_val(idx)

    @staticmethod
    def build_from_pron_len_to_long_seg_spans(pron_len_to_long_seg_spans):

        lss_list = []
        dic = {}
        for pron_len in range(1, max(pron_len_to_long_seg_spans.keys()) + 1):
            prons = pron_len_to_long_seg_spans.get(pron_len, [])
            range_list = []
            for _lss_list in prons:
                assert _lss_list
                lss_list += _lss_list
                range_list.append(len(_lss_list))
            dic[pron_len] = range_list
        cnr_index = CompactNumericRangeIndex.build_from_dict(dic)

        ls_num_arr = [lss.long_seg_num for lss in lss_list]
        ls_span_idx_arr = [lss.span_idx for lss in lss_list]

        dic = {}
        for pron_len in range(1, max(pron_len_to_long_seg_spans.keys()) + 1):
            dic[pron_len] = []
            for range_idx in cnr_index.iter_range_idx(pron_len):
                if cnr_index.get_num_val(range_idx) > 1:
                    dic[pron_len].append(range_idx)
        cn_index = CompactNumericIndex.build_from_dict(dic)

        return LSSIndex(
            cnr_index=cnr_index,
            cn_index=cn_index,
            ls_num_arr=np.array(ls_num_arr).astype(np.int32),
            ls_span_idx_arr=np.array(ls_span_idx_arr).astype(np.int32),
        )

    @staticmethod
    def from_fobj(fobj):
        cnr_index = CompactNumericRangeIndex.from_fobj(fobj)
        cn_index = CompactNumericIndex.from_fobj(fobj)
        ls_num_arr = pickle.load(fobj)
        ls_span_idx_arr = pickle.load(fobj)
        return LSSIndex(
            cnr_index=cnr_index,
            cn_index=cn_index,
            ls_num_arr=np.array(ls_num_arr).astype(np.int32),
            ls_span_idx_arr=np.array(ls_span_idx_arr).astype(np.int32),
        )

    def to_fobj(self, fobj):
        self.cnr_index.to_fobj(fobj)
        self.cn_index.to_fobj(fobj)
        pickle.dump(self.ls_num_arr, fobj)
        pickle.dump(self.ls_span_idx_arr, fobj)
