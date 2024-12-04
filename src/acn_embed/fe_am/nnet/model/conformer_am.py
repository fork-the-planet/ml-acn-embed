import torch
import torch.nn.functional
from torch import nn
from torchaudio.models import Conformer

from acn_embed.fe_am.nnet.model.base_am import BaseAm
from acn_embed.util.base_inference_input import BaseInferenceInput
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class ConformerAm(BaseAm):
    def __init__(
        self,
        *,
        dim_in,
        dim_out,
        frame_ms,
        num_subsample_layers,
        embedder_dim,
        num_heads,
        num_layers,
        conv_kernel_size,
    ):
        super().__init__(dim_in=dim_in, dim_out=dim_out, frame_ms=frame_ms)

        assert num_subsample_layers == 0  # Not supported in this implementation
        self.input_transform = nn.Linear(in_features=dim_in, out_features=embedder_dim)
        self._min_input_len = 1
        self.conformer = Conformer(
            input_dim=embedder_dim,
            num_heads=num_heads,
            ffn_dim=4 * embedder_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=conv_kernel_size,
            use_group_norm=True,
        )
        self.output_layer = nn.Linear(embedder_dim, dim_out)

    def forward(self, for_input: BaseInferenceInput):
        # We can't let this happen, else we'll get NaN
        if torch.any(for_input.input_len_t < self.min_input_len).item():
            raise RuntimeError(f"Got an input shorter than {self.min_input_len}")

        output = self.input_transform(for_input.input_t)
        assert output.shape[1] == for_input.input_len_t.max().item()  # sanity check
        output, output_lengths = self.conformer(output, for_input.input_len_t)
        output = self.output_layer(output)
        return output, output_lengths
