from abc import ABC

from torch import nn


class BaseEmbedder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    def min_input_len(self):
        """
        The minimum number of frames required in inputs
        """
        return self._min_input_len

    def get_metrics(self):
        return {}

    def get_log_str(self):
        return ""
