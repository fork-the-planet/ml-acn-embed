from acn_embed.embed.model.base.lstm_embedder import LstmEmbedder
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class FLstmEmbedder(LstmEmbedder):

    def __init__(
        self,
        *,
        dim_in,
        dim_out,
        dim_state,
        dropout_prob,
        num_layers,
    ):
        super().__init__(
            dim_in=dim_in,
            dim_out=dim_out,
            dim_state=dim_state,
            dropout_prob=dropout_prob,
            normalize_input=True,
            normalize_output=False,
            num_layers=num_layers,
        )
        LOGGER.info(f"Initialized {self.__str__()}")

    def __str__(self):
        return (
            f"[FLstmEmbedder "
            f"dim_in={self.dim_in} "
            f"dim_state={self.dim_state} "
            f"dim_out={self.dim_out} "
            f"num_layers={self.num_layers}]"
        )
