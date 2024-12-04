from functools import reduce
from operator import mul

from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class CombIterator:
    """
    Iterator that generates all possible combinations of items from a list of lists
    For example, given:
        [1, 2, 3],
        ['a'],
        [-10, -20]
    We generate:
        [1, 'a', -10],
        [2, 'a', -10],
        [3, 'a', -10],
        [1, 'a', -20],
        [2, 'a', -20],
        [3, 'a', -20],
    """

    def __init__(self, param_lists):
        """
        The position of the iterator is stored by "coordinates".
        e.g. coordinates [x, y, z] points to the combination
        [param_lists[0][x], param_lists[1][y], param_lists[2][z]]
        """
        self.param_lists = param_lists
        self.num_lists = len(param_lists)
        self.lens = [len(_list) for _list in param_lists]
        self.tot_num_combs = reduce(mul, self.lens, 1)
        self.coord = None

    def __iter__(self):
        return self

    def __next__(self):
        """
        We "increment" the coordinates to go to the next combination
        """
        if self.coord is None:
            self.coord = [0] * self.num_lists
        else:
            # Find the last dimension that can be incremented
            last_dim = 0
            while (last_dim < self.num_lists) and (
                self.coord[last_dim] == self.lens[last_dim] - 1
            ):
                self.coord[last_dim] = 0
                last_dim += 1
            if last_dim == self.num_lists:
                raise StopIteration
            self.coord[last_dim] += 1

        return [self.param_lists[dim][self.coord[dim]] for dim in range(self.num_lists)]
