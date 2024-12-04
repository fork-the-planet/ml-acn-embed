from acn_embed.embed.model.base.lstm_embedder import LstmEmbedder
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class GLstmEmbedder(LstmEmbedder):

    def __init__(
        self,
        *,
        dim_in,
        dim_out,
        dim_state,
        dropout_prob,
        num_layers,
        idx_to_subword,
    ):
        super().__init__(
            dim_in=dim_in,
            dim_out=dim_out,
            dim_state=dim_state,
            dropout_prob=dropout_prob,
            normalize_input=True,
            normalize_output=True,
            num_layers=num_layers,
        )
        assert len(idx_to_subword) == dim_in
        self.idx_to_subword = idx_to_subword
        self.subword_to_idx = {_subword: _idx for (_idx, _subword) in enumerate(idx_to_subword)}
        assert len(self.idx_to_subword) == len(self.subword_to_idx)
        LOGGER.info(f"Initialized {self.__str__()}")

    def __str__(self):
        return (
            f"[GLstmEmbedder "
            f"dim_in={self.dim_in} "
            f"dim_state={self.dim_state} "
            f"dim_out={self.dim_out} "
            f"num_layers={self.num_layers}]"
        )

    @property
    def num_subwords(self):
        return len(self.idx_to_subword)
