from abc import ABC, abstractmethod


class BaseDataloaderOutput(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def to(self, device):
        """
        Change device for all tensors stored in this object
        """
        raise NotImplementedError()
