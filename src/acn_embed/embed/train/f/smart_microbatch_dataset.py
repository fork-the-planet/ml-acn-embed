import gzip
import json
import os
import pickle
import random

import numpy as np
import torch
from torch.utils.data import Dataset

import acn_embed.util.torchutils
from acn_embed.embed.data.microbatch.index_overlapping_segments import LongSegSpan
from acn_embed.embed.data.microbatch.lss_index import LSSIndex
from acn_embed.embed.data.microbatch.span_info import SpanInfo
from acn_embed.util.base_trainer.base_dataloader_output import BaseDataloaderOutput
from acn_embed.util.data.storage import H5FeatStoreReader
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class SmartMicroBatchDatasetOutput(BaseDataloaderOutput):
    def __init__(
        self,
        *,
        input_t: torch.Tensor,
        input_len_t: torch.Tensor,
        fdiff_ref_t: torch.Tensor,
        microbatch_size: int,
        num_same: list[int],
    ):
        super().__init__()
        self.input_t = input_t
        self.input_len_t = input_len_t
        self.fdiff_ref_t = fdiff_ref_t
        self.microbatch_size = microbatch_size
        self.num_same = num_same

    def to(self, device):
        return SmartMicroBatchDatasetOutput(
            input_t=self.input_t.to(device=device),
            input_len_t=self.input_len_t,  # This must be left on CPU for pack_padded_sequence
            fdiff_ref_t=self.fdiff_ref_t.to(device=device),
            microbatch_size=self.microbatch_size,
            num_same=self.num_same,
        )


# pylint: disable=too-many-instance-attributes
class SmartMicroBatchDataset(Dataset):
    def __init__(
        self,
        *,
        data_path,
        len_microbatch,
        max_same,
        min_input_len,
        num_phones_to_prior,  # Either a dict (int->float) or a string (filename)
        tot_num_pivots,
        divide_by=1,
        min_len_pron=1,
    ):

        self.max_same = max_same
        self.len_microbatch = len_microbatch
        self.min_len_pron = min_len_pron
        self.min_input_len = min_input_len

        if isinstance(num_phones_to_prior, str):
            with open(num_phones_to_prior, "r", encoding="utf8") as fobj:
                num_phones_to_prior = {
                    int(key): float(val) for key, val in json.load(fobj).items()
                }

        assert isinstance(num_phones_to_prior, dict)

        # Normalize num_phones_to_prior so that priors sum to 1
        tot_prior = np.sum(list(num_phones_to_prior.values()))
        num_phones_to_prior = {
            length: (prior / tot_prior) for (length, prior) in num_phones_to_prior.items()
        }

        self.feat_store_fn = os.path.join(data_path, "amfeat.h5")

        self.metadata_fn = os.path.join(data_path, "metadata.pkl.gz")
        with gzip.open(self.metadata_fn, "rb") as fobj:
            self.long_segments = pickle.load(fobj)

        with gzip.open(os.path.join(data_path, "index.pkl.gz"), "rb") as fobj:
            self.index = LSSIndex.from_fobj(fobj)

        self.len_to_span_info = {}
        for long_segment in self.long_segments:
            if long_segment.length not in self.len_to_span_info:
                self.len_to_span_info[long_segment.length] = SpanInfo(long_segment.length)

        # The pron lengths of all the pivots used in one epoch.
        # Each pron length corresponds to one randomly-generated microbatch
        # (for which the pivot has that pron length).
        # The length of this list equals the total number of microbatches per epoch.
        self.pivot_pron_lengths = []

        self.max_len_pron = min(max(num_phones_to_prior.keys()), self.index.max_pron_len)

        self.pron_length_distribution = []
        for length in range(min_len_pron, self.max_len_pron + 1):
            num_duplicates = (
                min(
                    int(tot_num_pivots * num_phones_to_prior.get(length, 0)),
                    self.index.get_num_duplicate_prons(length),
                )
                // divide_by
            )

            self.pron_length_distribution.append(num_duplicates)
            self.pivot_pron_lengths += [length] * num_duplicates

            LOGGER.info(f"{length=} num_pivots={num_duplicates}")

        random.shuffle(self.pivot_pron_lengths)

        LOGGER.info(
            f"Total num unique pivots={len(self.pivot_pron_lengths)} "
            f"hash={hash(tuple(self.pivot_pron_lengths))}"
        )

        self.feat_store = None

    def __len__(self):
        return len(self.pivot_pron_lengths)

    def __getitem__(self, idx):
        if self.feat_store is None:
            self.feat_store = H5FeatStoreReader(self.feat_store_fn)

        pivot_and_same_lss_list, compete_lss_list = self.get_lss_lists(idx)
        num_same = len(pivot_and_same_lss_list) - 1  # The pivot itself is excluded
        num_different = len(compete_lss_list)

        return (
            [self.get_nparr(lss) for lss in pivot_and_same_lss_list + compete_lss_list],
            [1.0 / num_same] * num_same + [0.0] * num_different,
            num_same,
        )

    def get_lss_lists(self, idx):

        pivot_pron_length = self.pivot_pron_lengths[idx]

        # Choose a pron index
        pivot_pron_idx = self.index.get_random_dup_pron_idx(pron_len=pivot_pron_length)

        # Get a list of candidates for that pron
        pivot_and_same_lss_list = self.index.get_random_lss_list(pivot_pron_idx, 1 + self.max_same)

        num_same = len(pivot_and_same_lss_list) - 1  # The pivot itself is excluded
        num_different = self.len_microbatch - len(pivot_and_same_lss_list)

        assert num_same + num_different + 1 == self.len_microbatch
        assert num_same >= 1
        assert num_different >= 1

        compete_pron_lengths = random.choices(
            range(self.min_len_pron, self.max_len_pron + 1),
            weights=self.pron_length_distribution,
            k=num_different,
        )

        compete_lss_list = []
        for compete_pron_length in compete_pron_lengths:
            competing_pron_idx = self.index.get_random_pron_idx(
                compete_pron_length, exclude_pron_idx=pivot_pron_idx
            )
            assert compete_pron_length != pivot_pron_length or competing_pron_idx != pivot_pron_idx
            lss = self.index.get_random_lss(pron_idx=competing_pron_idx)
            compete_lss_list.append(lss)

        # This should work:
        # assert len(same_lss_list) + len(compete_lss_list) == self.len_microbatch
        return pivot_and_same_lss_list, compete_lss_list

    def get_tokens(self, lss: LongSegSpan):
        long_segment = self.long_segments[lss.long_seg_num]
        span_info = self.len_to_span_info[long_segment.length]
        span = span_info.idx_to_span[lss.span_idx]
        tokens = long_segment.token_list[span.start : span.end]
        return tokens

    def get_nparr(self, lss: LongSegSpan):
        tokens = self.get_tokens(lss)
        nparr = self.feat_store.get_nparr(lss.long_seg_num)
        return nparr[tokens[0]["start_frame"] : tokens[-1]["end_frame"]]

    def collate(self, batch):
        microbatch_size = None
        num_frames = []
        in_arrs = []
        fdiff_ref = []
        num_same = []

        LOGGER.debug("len(batch)=%s", len(batch))

        for _in_arr_list, _fdiff_ref_list, _num_positive in batch:
            if microbatch_size is None:
                microbatch_size = len(_in_arr_list)
            assert microbatch_size == len(_in_arr_list)

            for inarr in _in_arr_list:
                in_arrs.append(torch.tensor(inarr, dtype=torch.float32))
                num_frames.append(inarr.shape[0])

            fdiff_ref.append(_fdiff_ref_list)

            num_same.append(_num_positive)

        # Since each batch in turn contains microbatches (not utterances),
        # we have to "flatten" the batches
        # (batch_size * microbatch_size, num_frames_t.max(), dim_in)
        # Each elem is an integer storing the number of frames in each sequence
        # (batch_size*microbatch_size,)
        padded_input_t, num_frames_t = acn_embed.util.torchutils.pad_sequence_batch_first(
            sequences=in_arrs, min_len=self.min_input_len
        )

        # (batch_size, microbatch_size-1)
        fdiff_ref_t = torch.tensor(fdiff_ref, dtype=torch.float32)

        return SmartMicroBatchDatasetOutput(
            input_t=padded_input_t,
            input_len_t=num_frames_t,
            fdiff_ref_t=fdiff_ref_t,
            microbatch_size=microbatch_size,
            num_same=num_same,
        )
