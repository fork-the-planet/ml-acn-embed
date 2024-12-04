import torch


class BaseInferenceInput:
    def __init__(
        self,
        input_t: torch.Tensor,
        input_len_t: torch.Tensor,
    ):
        super().__init__()
        self.input_t = input_t
        self.input_len_t = input_len_t

    def to(self, device):
        return BaseInferenceInput(
            input_t=self.input_t.to(device), input_len_t=self.input_len_t.to(device)
        )
