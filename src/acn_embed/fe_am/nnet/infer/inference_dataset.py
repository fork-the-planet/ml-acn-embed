import torch
from torch.utils.data import Dataset

from acn_embed.util import torchutils
from acn_embed.util.data.storage import H5FeatStoreReader
from acn_embed.util.data.transcription import read_tran_utts
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class InferenceDataset(Dataset):
    def __init__(self, h5_fn, tran_fn, sort_utt_ids, min_input_len):

        self.h5_fn = h5_fn
        self.tran_fn = tran_fn
        self.h5_reader = None

        self.mdatas = list(read_tran_utts(tran_fn))

        if sort_utt_ids:
            self.dset_idx_to_internal_idx = sorted(
                range(len(self.mdatas)), key=lambda x: self.mdatas[x]["utt_id"]
            )
        else:
            self.dset_idx_to_internal_idx = list(range(len(self.mdatas)))

        # Get dims
        _h5_reader = H5FeatStoreReader(self.h5_fn)
        self.feat_dim = _h5_reader.get_nparr(0).shape[1]

        LOGGER.info(f"InferenceDataset len={len(self.mdatas)}")

        self.min_input_len = min_input_len

    def __len__(self):
        return len(self.mdatas)

    def __getitem__(self, dset_idx):
        if self.h5_reader is None:
            self.h5_reader = H5FeatStoreReader(self.h5_fn)

        h5_idx = self.dset_idx_to_internal_idx[dset_idx]

        return [self.h5_reader.get_nparr(h5_idx), self.mdatas[h5_idx]["utt_id"]]

    def collate(self, data):
        """
        Each member of batch has two elems: nparray and ref_ints
        """
        fbanks = []
        utt_ids = []
        for nparray, utt_id in data:
            fbanks.append(torch.tensor(nparray, dtype=torch.float32))
            utt_ids.append(utt_id)

        fbanks, input_lengths = torchutils.pad_sequence_batch_first(
            fbanks, min_len=self.min_input_len
        )

        return (fbanks, utt_ids, input_lengths)
