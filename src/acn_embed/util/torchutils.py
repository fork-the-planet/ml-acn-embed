import torch
import torch.nn

from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def get_num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pad_sequence_batch_first(sequences: list, min_len, min_padded_len=None):
    """
    Same as torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.0) but
    enforces a minimum sequence length, as well as an optional minimum padded length
    Args:
        min_len:
            Ensures each sequence is at least this length, padding with zeros as necessary, and
            includes the zeros as part of the sequence. This will change the returned sequence
            length.
        min_padded_len:
            If given, ensures that the sequences are zero-padded to achieve this total
            length. This does not change the returned sequence length.
    Returns:
        Zero-padded sequences
        Lengths of the sequences
    """
    # Tensor must be in CPU for pack_padded_sequence
    len_t = torch.tensor(
        [max(min_len, seq.shape[0]) for seq in sequences],
        dtype=torch.int32,
        device=torch.device("cpu"),
    )
    max_len = max(len_t)
    padded_len = max_len

    if min_padded_len is not None:
        padded_len = max(max_len, min_padded_len)

    padded_seqs = torch.zeros(
        *([len(sequences), padded_len] + list(sequences[0].shape)[1:]),
        dtype=sequences[0].dtype,
        device=sequences[0].device,
    )

    for batch, seq in enumerate(sequences):
        padded_seqs[batch, : seq.shape[0]] = seq

    return padded_seqs, len_t
