from pathlib import Path

import numpy as np
import torch

from acn_embed.embed.model.base.meanvar_normalizer import MeanVarNormalizer
from acn_embed.util.base_trainer import base_functions
from acn_embed.util.data.storage import H5FeatStoreReader
from acn_embed.util.utils import get_start_end_idx


def get_meanvar_normalizer_dist(data_sum, data_sq_sum, data_size):
    """
    Aggregate sums and sums of squares from distributed nodes to return a
    MeanVarNormalizer
    """
    # pylint: disable=unbalanced-tuple-unpacking
    data_sum, data_sq_sum, data_size = base_functions.dist_get_summed(
        data_sum, data_sq_sum, data_size
    )
    normalizer = MeanVarNormalizer(data_sum.shape[-1])
    normalizer.set_from_mv_t(
        mean_t=data_sum / data_size,
        std_t=torch.sqrt(data_sq_sum / data_size - torch.pow(data_sum / data_size, 2)),
    )
    return normalizer


def get_input_meanvar_normalizer_dist(h5_path: Path, num_splits: int, split: int):
    assert 1 <= split <= num_splits
    reader = H5FeatStoreReader(path=h5_path)
    arr_sum = None
    arr2_sum = None
    n = 0
    dim = None
    start_idx, end_idx = get_start_end_idx(reader.size, num_splits, split)
    for idx in range(start_idx, end_idx):
        arr = reader.get_nparr(idx)
        assert arr.ndim == 2
        if dim is None:
            dim = arr.shape[1]
        assert arr.shape[1] == dim
        assert np.all(np.isfinite(arr))
        n += arr.shape[0]
        _arr_sum = np.sum(arr, axis=0)
        _arr2_sum = np.sum(np.power(arr, 2), axis=0)
        if arr_sum is None:
            arr_sum = _arr_sum
        else:
            arr_sum += _arr_sum
        if arr2_sum is None:
            arr2_sum = _arr2_sum
        else:
            arr2_sum += _arr2_sum
    return get_meanvar_normalizer_dist(data_sum=arr_sum, data_sq_sum=arr2_sum, data_size=n)
