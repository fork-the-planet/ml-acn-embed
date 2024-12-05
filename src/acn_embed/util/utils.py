import gzip

from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def count_lines(filename):
    """
    Count the number of lines in a text or gzipped text file
    """
    filename = str(filename)
    with (
        gzip.open(filename, "rt", encoding="utf8")
        if filename.endswith(".gz")
        else open(filename, "r", encoding="utf8")
    ) as fobj:
        return len(fobj.readlines())


def get_start_end_idx(total, num_splits, split):
    """
    Assuming a set of "total" items, split into "num_splits" sets and
    return the start and ending index of the "split"'th set.
    The start index is inclusive, whereas the end index is exclusive.
    """
    assert 1 <= split <= num_splits
    assert total >= num_splits

    if split == 1:
        start_idx = 0
    else:
        start_idx = int((split - 1) * total / num_splits)

    if split == num_splits:
        end_idx = total
    else:
        end_idx = int(split * total / num_splits)

    return start_idx, end_idx
