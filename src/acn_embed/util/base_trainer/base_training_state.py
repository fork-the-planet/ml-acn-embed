class BaseTrainingState:

    # pylint: disable=too-many-instance-attributes
    def __init__(self):
        self.best_epoch = None  # The epoch that produced the best model
        self.best_epoch_tr_loss = None  # The TR loss that occurred in best_epoch
        self.best_model_fn = ""  # The file storing the best model so far
        self.cv_loss = None
        self.epoch = 0
        self.epoch_time = None
        self.lrate = None
        self.min_cv_loss = None  # The minimum CV loss seen so far
        self.model_fn = None
        self.tr_loss = None

    def metrics(self):
        return {
            "cv-loss": self.cv_loss,
            "tr-loss": self.tr_loss,
            "learning-rate": self.lrate,
            "epoch-time": self.epoch_time,
        }

    def log_str(self):
        return f"tr-loss={self.tr_loss:.5f} cv-loss={self.cv_loss:.5f} lrate={self.lrate:.3e}"
