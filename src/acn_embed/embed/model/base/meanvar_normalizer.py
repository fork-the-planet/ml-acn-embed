import torch
from torch import nn

from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class MeanVarNormalizer(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self._dim = dim
        # Initialize with random Parameters with the correct size,
        # so that load_state_dict() will work
        self.normalizer_scale = nn.parameter.Parameter(
            torch.empty((dim,), dtype=torch.float32), requires_grad=False
        )
        self.normalizer_offset = nn.parameter.Parameter(
            torch.empty((dim,), dtype=torch.float32), requires_grad=False
        )

    def set_from_mv_t(self, mean_t: torch.Tensor, std_t: torch.Tensor):
        assert mean_t.ndim == 1
        assert std_t.ndim == 1
        assert mean_t.numel() == self._dim
        assert std_t.numel() == self._dim
        std_t[std_t < 1.0e-8] = 1.0e-8
        normalizer_scale = 1.0 / std_t
        normalizer_offset = -normalizer_scale * mean_t
        assert torch.all(torch.isfinite(normalizer_scale))
        assert torch.all(torch.isfinite(normalizer_offset))
        assert normalizer_scale.ndim == 1
        assert normalizer_offset.ndim == 1
        self.normalizer_scale = nn.parameter.Parameter(
            normalizer_scale.to(dtype=torch.float32), requires_grad=False
        )
        self.normalizer_offset = nn.parameter.Parameter(
            normalizer_offset.to(dtype=torch.float32), requires_grad=False
        )

    @property
    def dim(self):
        return self._dim

    def forward(self, input_t: torch.Tensor, denormalize: bool):
        if denormalize:
            return (input_t - self.normalizer_offset) / self.normalizer_scale
        return input_t * self.normalizer_scale + self.normalizer_offset
