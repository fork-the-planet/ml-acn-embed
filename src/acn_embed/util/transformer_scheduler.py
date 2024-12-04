import numpy as np
import numpy.testing

from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


# pylint: disable=duplicate-code
class TransformerScheduler:
    def __init__(self, optimizer, peak_lr, warmup_steps_k):
        self._optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = int(warmup_steps_k * 1000)
        self._scale = peak_lr * np.sqrt(self.warmup_steps)
        self._steps_completed = 0
        LOGGER.info(
            f"Initialized TransformerScheduler with "
            f"{peak_lr=:.3e} "
            f"warmup_steps_k={self.warmup_steps / 1000} "
        )
        self._lr = 0.0
        self.step()  # Take a step here to skip lr==0 (which does nothing)

    def step(self):
        self._steps_completed += 1
        self.update_lr()

    def update_lr(self):
        if self._steps_completed <= self.warmup_steps:
            mult = np.power(self.warmup_steps, -1.5) * self._steps_completed
        else:
            mult = 1.0 / np.sqrt(self._steps_completed)

        lr = self._scale * mult

        # sanity check
        if self._steps_completed == self.warmup_steps:
            numpy.testing.assert_almost_equal(lr, self.peak_lr)

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

        self._lr = lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != "_optimizer"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self.update_lr()

    @property
    def steps_completed(self):
        return self._steps_completed

    def get_last_lr(self):
        return self._lr
