import torch
import torch.nn.functional
from torch import nn

from acn_embed.embed.model.base.base_embedder import BaseEmbedder
from acn_embed.embed.model.base.meanvar_normalizer import MeanVarNormalizer
from acn_embed.util.base_inference_input import BaseInferenceInput


class LstmEmbedder(BaseEmbedder):

    def __init__(
        self,
        *,
        dim_in,
        dim_out,
        dim_state,
        dropout_prob,
        normalize_input: bool,
        normalize_output: bool,
        num_layers,
    ):
        super().__init__()

        if num_layers == 1:
            dropout_prob = 0.0

        self.lstm = nn.LSTM(
            input_size=dim_in,
            hidden_size=dim_state,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout_prob,
            batch_first=True,  # Input must be (batch, seq, feature)
        )

        self.output_nnet = nn.Linear(2 * dim_state, dim_out)

        self.in_normalizer = None
        self.out_normalizer = None
        if normalize_input:
            self.in_normalizer = MeanVarNormalizer(dim_in)
        if normalize_output:
            self.out_normalizer = MeanVarNormalizer(dim_out)

        self._min_input_len = 1

        # Sanity check
        assert dim_in == self.lstm.input_size
        assert dim_state == self.lstm.hidden_size
        assert num_layers == self.lstm.num_layers

    @property
    def dim_in(self):
        return self.lstm.input_size

    @property
    def dim_out(self):
        return self.output_nnet.out_features

    @property
    def dim_state(self):
        return self.lstm.hidden_size

    @property
    def num_layers(self):
        return self.lstm.num_layers

    def forward(self, for_input: BaseInferenceInput):
        """
        Note: for_input.input_len_t must be list or CPU tensor for pack_padded_sequence
        If set to None, we assume that all sequences have the same length
        (the number of columns in padded_input), and use the entirety of padded_input
        as-is

        Returns:
            A tensor with shape (num_sequences, dim_out)
        """
        input_t = for_input.input_t
        if self.in_normalizer:
            input_t = self.in_normalizer.forward(for_input.input_t, denormalize=False)

        self.lstm.flatten_parameters()
        if for_input.input_len_t is None:
            output_t, _ = self.lstm(input_t)
            output_lengths = [output_t.shape[1]] * output_t.shape[0]

        else:
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(
                input_t,
                for_input.input_len_t.to(device=torch.device("cpu")),
                batch_first=True,
                enforce_sorted=False,
            )
            output_t, _ = self.lstm(packed_input)

            output_t, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                output_t, batch_first=True, total_length=torch.max(for_input.input_len_t).item()
            )

        num_sequences = output_t.shape[0]

        final_outputs = torch.empty(
            num_sequences, 2 * self.dim_state, device=for_input.input_t.device
        )
        for idx in range(num_sequences):
            final_outputs[idx, : self.dim_state] = output_t[
                idx, output_lengths[idx] - 1, : self.dim_state
            ]
            final_outputs[idx, self.dim_state :] = output_t[idx, 0, self.dim_state :]

        final_outputs = self.output_nnet(final_outputs)

        if self.out_normalizer:
            final_outputs = self.out_normalizer.forward(final_outputs, denormalize=True)

        return final_outputs
