from pathlib import Path

import h5py
import numpy as np

from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def get_feat_dim(path: Path):
    reader = H5FeatStoreReader(path)
    return reader.get_nparr(0).shape[-1]


class H5FeatStoreReader:
    def __init__(self, path: Path):
        self.path = path
        self.h5_obj = h5py.File(str(self.path), "r")
        if not self.h5_obj.keys():
            self.max_group = -1
            self.max_num = -1
        else:
            self.max_group = max(int(key) for key in self.h5_obj.keys())
            self.max_num = max(int(key) for key in self.h5_obj[str(self.max_group)].keys())
        LOGGER.debug(f"Initialized h5 from {path} max_num={self.max_num=}")

    def get_nparr(self, num):
        assert num <= self.max_num
        group_num = num // 1000000
        assert group_num <= self.max_group
        return self.h5_obj[str(group_num)][str(num)][:]

    def close(self):
        self.h5_obj.close()

    @property
    def size(self):
        return self.max_num + 1


class H5FeatStoreWriter:
    def __init__(self, path, compress):
        self.path = path
        self.h5_obj = h5py.File(str(self.path), "w")
        self.compress = compress
        self._num_written = 0

    def add_nparr(self, num, nparr):
        group_num = num // 1000000
        group = self.h5_obj.require_group(str(group_num))
        if self.compress:
            group.create_dataset(str(num), data=nparr.astype(np.float32), compression="gzip")
        else:
            group.create_dataset(str(num), data=nparr.astype(np.float32))
        self._num_written += 1

    @property
    def num_written(self):
        return self._num_written

    def close(self):
        LOGGER.info(f"Wrote {self.path} with {self._num_written} entries added")
        self.h5_obj.close()
