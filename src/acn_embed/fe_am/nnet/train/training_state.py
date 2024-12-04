from acn_embed.util.base_trainer.base_training_state import BaseTrainingState


class TrainingState(BaseTrainingState):
    def __init__(self):
        super().__init__()
        self.best_prior_fn = ""
