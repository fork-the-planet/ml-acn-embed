import gzip
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset

import acn_embed.util.torchutils
from acn_embed.util.data.storage import H5FeatStoreReader
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class SegmentAmFeatDataset(Dataset):
    def __init__(self, h5_path: Path, segment_path: Path, min_input_len):
        self.feat_reader = H5FeatStoreReader(h5_path)
        with gzip.open(segment_path, "rb") as fobj:
            self.segments = pickle.load(fobj)
        self.min_input_len = min_input_len
        LOGGER.info(f"{len(self.segments)=}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return idx, self.get_nparr(idx)

    def get_nparr(self, segment_idx):
        segment = self.segments[segment_idx]
        nparr = self.feat_reader.get_nparr(segment["long_segment_num"])
        return nparr[segment["start_frame"] : segment["end_frame"]]

    def collate(self, batch):
        idxlist = [item[0] for item in batch]
        arrlist = [torch.tensor(item[1]) for item in batch]
        num_sequences = len(arrlist)
        padded_input_t, num_frames_t = acn_embed.util.torchutils.pad_sequence_batch_first(
            sequences=arrlist, min_len=self.min_input_len
        )
        return idxlist, padded_input_t, num_sequences, num_frames_t
