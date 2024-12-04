import gzip
import pickle

import kaldi_io
import torch
from torch.utils.data import Dataset

import acn_embed.util.data.transcription
from acn_embed.util import torchutils
from acn_embed.util.base_trainer.base_dataloader_output import BaseDataloaderOutput
from acn_embed.util.data.storage import H5FeatStoreReader
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class TrainingDatasetOutput(BaseDataloaderOutput):
    def __init__(
        self,
        input_t: torch.Tensor,
        ref_t: torch.Tensor,
        input_len_t: torch.Tensor,
        ref_len_t: torch.Tensor,
    ):
        super().__init__()
        self.input_t = input_t
        self.ref_t = ref_t
        self.input_len_t = input_len_t
        self.ref_len_t = ref_len_t

    def to(self, device):
        return TrainingDatasetOutput(
            input_t=self.input_t.to(device),
            ref_t=self.ref_t.to(device),
            input_len_t=self.input_len_t.to(device),
            ref_len_t=self.ref_len_t.to(device),
        )


def get_utt_to_pdfs(pdf_ark_gz_fn):
    with gzip.open(pdf_ark_gz_fn, "rb") as fobj:
        # Need to copy() to avoid the error:
        # "ValueError: given numpy array strides not a multiple of the element byte size.
        # Copy the numpy array to reallocate the memory."
        return {utt: nparr.copy() for utt, nparr in kaldi_io.read_vec_int_ark(fobj)}


class TrainingDataset(Dataset):
    def __init__(
        self,
        *,
        h5_fn,
        tran_fn,
        min_input_len,
        max_input_len=3000,
        pdf_ark_gz_fn=None,
        utt_to_pdfs_fn=None,
    ):

        self.h5_fn = h5_fn
        self.tran_fn = tran_fn
        self.min_input_len = min_input_len

        self.h5_reader = None

        if pdf_ark_gz_fn:
            assert utt_to_pdfs_fn is None
            utt_to_pdfs = get_utt_to_pdfs(pdf_ark_gz_fn)
        else:
            assert utt_to_pdfs_fn
            with open(utt_to_pdfs_fn, "rb") as fobj:
                utt_to_pdfs = pickle.load(fobj)

        self.refs = []

        # Stores a map from dataset idx (externally exposed) -> internal "original" idx for
        # use with h5 file.
        # We need this because not all the data will be used (pdfs might be missing, etc.)
        self.dset_idx_to_orig_idx = []
        orig_idx = 0

        for orig_idx, utt in enumerate(acn_embed.util.data.transcription.read_tran_utts(tran_fn)):

            utt_id = utt["utt_id"]

            if utt_id not in utt_to_pdfs:
                continue

            pdfs = utt_to_pdfs.get(utt_id)

            if pdfs.size == 0:
                continue

            if max_input_len is not None and pdfs.size > max_input_len:
                LOGGER.warning(f"Discarding long utt {utt_id=} {pdfs.size=}")
                continue

            self.refs.append(pdfs)
            self.dset_idx_to_orig_idx.append(orig_idx)

        # Get dims
        _h5_reader = H5FeatStoreReader(self.h5_fn)
        self.feat_dim = _h5_reader.get_nparr(0).shape[1]

        assert orig_idx == _h5_reader.max_num

        if len(self.dset_idx_to_orig_idx) == 0:
            raise RuntimeError(
                "Dataset size is 0. You most likely supplied a wrong transcription file."
            )

        LOGGER.info(
            f"TrainingDataset len={len(self.dset_idx_to_orig_idx)} feat_dim={self.feat_dim} "
            f"orig_size={orig_idx + 1}"
        )

    def __len__(self):
        return len(self.dset_idx_to_orig_idx)

    def __getitem__(self, dset_idx):
        if self.h5_reader is None:
            self.h5_reader = H5FeatStoreReader(self.h5_fn)

        nparr = self.h5_reader.get_nparr(self.dset_idx_to_orig_idx[dset_idx])
        refs = self.refs[dset_idx]

        return [nparr, refs]

    def collate(self, data):
        ref_ids = []
        ref_lengths = []
        fbanks = []
        for nparray, ref_ints in data:
            fbanks.append(torch.from_numpy(nparray).to(dtype=torch.float32))
            ref_ids.append(
                torch.from_numpy(ref_ints).to(dtype=torch.long)
            )  # CrossEntropyLoss requires Long
            ref_lengths.append(len(ref_ints))

        fbanks, input_lengths_t = torchutils.pad_sequence_batch_first(
            fbanks, min_len=self.min_input_len
        )
        ref_ids, _ = torchutils.pad_sequence_batch_first(ref_ids, min_len=self.min_input_len)

        return TrainingDatasetOutput(
            input_t=fbanks,
            ref_t=ref_ids,
            input_len_t=input_lengths_t,
            ref_len_t=torch.tensor(ref_lengths, dtype=torch.int32),
        )
