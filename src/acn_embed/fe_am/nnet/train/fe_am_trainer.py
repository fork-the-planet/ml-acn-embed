import os
import shutil

import torch
import yaml
from torch import optim

import acn_embed.fe_am.nnet.model.get_model
from acn_embed.fe_am.nnet.train.training_dataset import TrainingDataset, TrainingDatasetOutput
from acn_embed.fe_am.nnet.train.training_state import TrainingState
from acn_embed.util.base_trainer import base_functions
from acn_embed.util.base_trainer.base_inference_output import BaseInferenceOutput
from acn_embed.util.base_trainer.base_trainer import BaseTrainer
from acn_embed.util.logger import get_logger
from acn_embed.util.torchutils import get_num_trainable_params
from acn_embed.util.transformer_scheduler import TransformerScheduler

LOGGER = get_logger(__name__)


class InferenceOutput(BaseInferenceOutput):
    def __init__(self, output_t: torch.Tensor, output_len_t: torch.Tensor):
        super().__init__()
        self.output_t = output_t
        self.output_len_t = output_len_t


class FeAmTrainer(BaseTrainer):

    def __init__(
        self,
        args,
        model_spec,
        scheduler_spec,
    ):

        super().__init__(args)

        self.model_spec = model_spec
        self.scheduler_spec = scheduler_spec

        base_functions.log_local_leader(
            "info", "model_spec=\n" + yaml.safe_dump(model_spec, sort_keys=False)
        )
        base_functions.log_local_leader(
            "info", "scheduler_spec=\n" + yaml.safe_dump(scheduler_spec, sort_keys=False)
        )

        self.loss_fun = torch.nn.CrossEntropyLoss(weight=None, reduction="sum")

    def get_datasets(self):
        rank = base_functions.get_rank()
        dataset_tr = TrainingDataset(
            h5_fn=os.path.join(self.args.data_path_tr, f"fbank.{rank}.h5"),
            tran_fn=os.path.join(self.args.data_path_tr, f"tran.{rank}.jsonl.gz"),
            min_input_len=self.model.min_input_len,
            pdf_ark_gz_fn=os.path.join(self.args.data_path_tr, f"pdf.{rank}.ark.gz"),
        )
        dataset_cv = TrainingDataset(
            h5_fn=os.path.join(self.args.data_path_cv, f"fbank.{rank}.h5"),
            tran_fn=os.path.join(self.args.data_path_cv, f"tran.{rank}.jsonl.gz"),
            min_input_len=self.model.min_input_len,
            pdf_ark_gz_fn=os.path.join(self.args.data_path_cv, f"pdf.{rank}.ark.gz"),
        )
        return dataset_tr, dataset_cv

    def get_init_model(self) -> torch.nn.Module:
        model = acn_embed.fe_am.nnet.model.get_model.get(model_spec=self.model_spec).to(
            self.device
        )
        base_functions.log_local_leader(
            "info", f"Trainable model parameters {get_num_trainable_params(model):,d}"
        )
        return model

    def get_new_training_state(self) -> TrainingState:
        return TrainingState()

    def get_optimizer(self) -> torch.optim.Optimizer:
        return optim.AdamW(
            self.model.parameters(), lr=0.1, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.001
        )

    def get_scheduler(self):
        return TransformerScheduler(self.optimizer, **self.scheduler_spec)

    def infer_1batch(self, dataloader_out: TrainingDatasetOutput) -> InferenceOutput:
        output_t, output_len_t = self.model(dataloader_out)
        return InferenceOutput(output_t=output_t, output_len_t=output_len_t)

    def compute_loss_1batch(
        self, dataloader_out: TrainingDatasetOutput, inference_out: InferenceOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # These should match because we don't do frame subsampling
        assert torch.all(inference_out.output_len_t == dataloader_out.ref_len_t)
        loss_sum = None
        for utt in range(inference_out.output_t.shape[0]):
            _loss_sum = self.loss_fun(
                inference_out.output_t[utt, : inference_out.output_len_t[utt], :],
                dataloader_out.ref_t[utt, : inference_out.output_len_t[utt]],
            )
            if utt == 0:
                loss_sum = _loss_sum
            else:
                loss_sum += _loss_sum
        tot_frames = torch.sum(inference_out.output_len_t)
        return loss_sum, tot_frames

    def handle_min_cv_loss(self):
        """
        Stuff to do when the best CV loss has been observed
        """
        self.ts.best_epoch = self.ts.epoch
        self.ts.min_cv_loss = self.ts.cv_loss
        self.ts.best_epoch_tr_loss = self.ts.tr_loss
        self.ts.best_prior_fn = self.compute_and_save_prior(self.ts.epoch)
        self.ts.best_model_fn = self.ts.model_fn
        base_functions.log_local_leader(
            "info", "Best cv-loss=%e model=%s", self.ts.min_cv_loss, self.ts.best_model_fn
        )
        base_functions.log_leader_metrics(
            metrics={
                "best-cv-loss": self.ts.min_cv_loss,
                "best-model-fn": self.ts.best_model_fn,
                "best-epoch": str(self.ts.best_epoch),
                "best-epoch-tr-loss": self.ts.best_epoch_tr_loss,
            }
        )
        if base_functions.is_leader() and self.args.model_output_leader_dir:
            best_model_fn = os.path.join(self.args.model_output_leader_dir, "model.best.pt")
            shutil.copyfile(self.ts.best_model_fn, best_model_fn)
            LOGGER.info(f"Updated {best_model_fn}")
            best_prior_fn = os.path.join(self.args.model_output_leader_dir, "prior.best.pt")
            shutil.copyfile(self.ts.best_prior_fn, best_prior_fn)
            LOGGER.info(f"Updated {best_prior_fn}")

        if base_functions.is_local_leader():
            best_model_fn = os.path.join(self.args.model_output_local_dir, "model.best.pt")
            shutil.copyfile(self.ts.best_model_fn, best_model_fn)
            LOGGER.info(f"Updated {best_model_fn}")
            best_prior_fn = os.path.join(self.args.model_output_local_dir, "prior.best.pt")
            shutil.copyfile(self.ts.best_prior_fn, best_prior_fn)
            LOGGER.info(f"Updated {best_prior_fn}")

    def compute_and_save_prior(self, epoch):
        prior_fn = os.path.join(self.args.model_output_local_dir, f"prior.{epoch:02d}.pt")
        output_log_sum_t, n = self.get_priors()
        if base_functions.is_local_leader():
            os.makedirs(self.args.model_output_local_dir, exist_ok=True)
            torch.save({"output_log_sum": output_log_sum_t, "n": n}, prior_fn)
            LOGGER.info(f"Dumped prior info to {prior_fn}")
        if base_functions.is_leader() and self.args.model_output_leader_dir:
            leader_prior_fn = os.path.join(
                self.args.model_output_leader_dir, f"prior.{epoch:02d}.pt"
            )
            os.makedirs(self.args.model_output_leader_dir, exist_ok=True)
            torch.save({"output_log_sum": output_log_sum_t, "n": n}, leader_prior_fn)
            LOGGER.info(f"Dumped prior info to {leader_prior_fn}")
        return prior_fn

    def get_priors(self):
        with torch.no_grad():
            log_softmax = torch.nn.LogSoftmax(dim=1)
            self.model.eval()
            output_log_sum = None
            n = 0
            for tr_dataloader_output in self.loader_tr:
                tr_dataloader_output = tr_dataloader_output.to(device=self.device)
                inference_out = self.infer_1batch(dataloader_out=tr_dataloader_output)
                for utt in range(inference_out.output_t.shape[0]):
                    _output = inference_out.output_t[utt, : inference_out.output_len_t[utt], :]
                    n += _output.shape[0]
                    _output_sum = torch.logsumexp(log_softmax(_output), dim=0)
                    if output_log_sum is None:
                        output_log_sum = _output_sum
                    else:
                        output_log_sum = torch.logaddexp(_output_sum, output_log_sum)

            output_log_sum_list = base_functions.dist_gather(dtype=torch.float32, x=output_log_sum)
            output_log_sum_t = torch.logsumexp(
                torch.cat([x.unsqueeze(0) for x in output_log_sum_list], dim=0), dim=0
            )
            base_functions.log_local_leader("info", f"n={n} output.shape={output_log_sum_t.shape}")
            self.model.train()
            return output_log_sum_t, n

    @staticmethod
    def add_args(parser):
        BaseTrainer.add_args(parser)
        parser.add_argument("--acoustic-feat-dim", action="store", type=int, required=True)
        parser.add_argument("--data-path-cv", action="store", type=str, required=True)
        parser.add_argument("--data-path-tr", action="store", type=str, required=True)
        parser.add_argument("--dim-out", action="store", type=int, default=None)
        parser.add_argument("--dim-out-fn", action="store", type=str, default=None)
        parser.add_argument("--max-lrate-mult", action="store", type=float, default=None)
        parser.add_argument(
            "--model-size",
            choices=["1M", "5M", "10M", "20M", "30M", "50M", "100M"],
            required=False,
        )
        parser.add_argument("--warmup-steps-k", action="store", type=float, default=None)
