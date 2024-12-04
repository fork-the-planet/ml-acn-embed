import pickle
from pathlib import Path

import numpy as np
import torch

from acn_embed.embed.train.f.normalize import get_meanvar_normalizer_dist
from acn_embed.embed.train.g.dataset import read_subwords
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def get_input_meanvar_normalizer_dist(
    segment_path: Path, subword_type: str, subword_str_to_id: dict
):
    subword_id_arr, _, _, _ = read_subwords(segment_path, subword_type, subword_str_to_id)
    embedding_sum = np.zeros((len(subword_str_to_id),), dtype=np.float32)
    for phoneid in subword_id_arr:
        embedding_sum[phoneid] += 1.0
    return get_meanvar_normalizer_dist(
        data_sum=embedding_sum, data_sq_sum=embedding_sum, data_size=subword_id_arr.size
    )


def get_output_meanvar_normalizer_dist(foutput_path: Path):
    with foutput_path.open("rb") as fobj:
        pickle.load(fobj)
        foutput = torch.from_numpy(pickle.load(fobj).astype(np.float32)).cuda()
    return get_meanvar_normalizer_dist(
        data_sum=torch.sum(foutput, dim=0),
        data_sq_sum=torch.sum(torch.pow(foutput, 2), dim=0),
        data_size=foutput.shape[0],
    )
