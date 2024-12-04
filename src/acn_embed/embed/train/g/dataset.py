import gzip
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import acn_embed.util.torchutils
from acn_embed.util.base_trainer.base_dataloader_output import BaseDataloaderOutput
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class MinibatchDatasetOutput(BaseDataloaderOutput):
    def __init__(
        self,
        input_t: torch.Tensor,
        ref_t: torch.Tensor,
        input_len_t: torch.Tensor,
    ):
        super().__init__()
        self.input_t = input_t
        self.ref_t = ref_t
        self.input_len_t = input_len_t

    def to(self, device):
        return MinibatchDatasetOutput(
            input_t=self.input_t.to(device),
            ref_t=self.ref_t.to(device),
            input_len_t=self.input_len_t,  # Always needs to be in CPU so don't move
        )


class MinibatchDataset(Dataset):
    def __init__(
        self,
        foutput_path: Path,
        segment_path: Path,
        subword_type: str,
        subword_str_to_id: dict,
    ):
        with foutput_path.open("rb") as fobj:
            pickle.load(fobj)  # all_foutput.shape not used
            all_foutput = pickle.load(fobj).astype(np.float32)
        self._dim_embedding = all_foutput.shape[1]
        self.subword_id_arr, self.data_id_to_subword_id, valid_idx, orig_len = read_subwords(
            segment_path, subword_type, subword_str_to_id
        )
        assert all_foutput.shape[0] == orig_len, "Length mismatch between foutput and segments"
        self.foutput_arr = all_foutput[valid_idx, :]
        self._num_subwords = len(subword_str_to_id)
        assert self.foutput_arr.shape[0] == self.data_id_to_subword_id.size

    def __len__(self):
        return self.foutput_arr.shape[0]

    def __getitem__(self, idx):
        subword_id_start_idx = self.data_id_to_subword_id[idx - 1] if idx > 0 else 0
        subword_id_end_idx = self.data_id_to_subword_id[idx]
        return (
            self.foutput_arr[idx, :],
            self.subword_id_arr[subword_id_start_idx:subword_id_end_idx],
        )

    @property
    def dim_embedding(self):
        return self._dim_embedding

    @property
    def num_subwords(self):
        return self._num_subwords

    def collate(self, batch):
        foutput_t = torch.cat(
            [torch.unsqueeze(torch.tensor(item[0]), dim=0) for item in batch], dim=0
        )
        assert foutput_t.shape == (len(batch), batch[0][0].size)

        # pylint: disable=not-callable
        padded_input_t, num_frames_t = acn_embed.util.torchutils.pad_sequence_batch_first(
            sequences=[
                torch.nn.functional.one_hot(
                    torch.LongTensor(item[1]), num_classes=self.num_subwords
                ).to(dtype=torch.float32)
                for item in batch
            ],
            min_len=1,
        )
        return MinibatchDatasetOutput(
            input_t=padded_input_t,
            ref_t=foutput_t,
            input_len_t=num_frames_t,
        )


def _get_subwords_from_metadata(metadata, subword_type):
    assert subword_type in ["grapheme", "phone"]
    if subword_type == "grapheme":
        return [list(mdata["orth"]) for mdata in metadata]
    return [mdata["pron"].split() for mdata in metadata]


def _get_subword_id_arr(subword_seqs, subword_str_to_id):
    """
    Returns a 1-d array of subword Ids, and a 1-d array of indices pointing to the
    first element of each subword sequence (except for the first, which is just 0)

    subword_seqs: 2-d list of subword sequences
    subword_str_to_id: dict mapping subword -> ID (0-based)
    """
    subword_id_arr = []
    _num_subwords = []
    for subword_seq in subword_seqs:
        _id_list = [subword_str_to_id[subword] for subword in subword_seq]
        subword_id_arr += _id_list
        _num_subwords.append(len(_id_list))
    subword_id_arr = np.array(subword_id_arr).astype(np.int16)
    data_id_to_subword_id = np.cumsum(_num_subwords).astype(np.int32)
    return subword_id_arr, data_id_to_subword_id


def read_subwords(segment_path: Path, subword_type: str, subword_str_to_id: dict):
    with gzip.open(segment_path, "rb") as fobj:
        metadata = pickle.load(fobj)
    subwords = _get_subwords_from_metadata(metadata, subword_type)
    valid_idx = [x for x in range(len(subwords)) if subwords[x]]
    subwords = [subwords[x] for x in valid_idx]
    subword_id_arr, data_id_to_subword_id = _get_subword_id_arr(subwords, subword_str_to_id)
    return subword_id_arr, data_id_to_subword_id, valid_idx, len(metadata)
