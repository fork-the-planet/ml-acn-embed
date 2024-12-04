from abc import ABC

from torch import nn


class BaseAm(nn.Module, ABC):
    def __init__(
        self,
        dim_in,
        dim_out,
        frame_ms,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.frame_ms = frame_ms

    def get_frame_ms(self):
        """
        Returns the time offset per frame in milliseconds
        """
        return self.frame_ms

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
