import os
import shutil
import time
from abc import abstractmethod

import torch

from acn_embed.util.base_trainer import base_functions
from acn_embed.util.base_trainer.base_dataloader_output import BaseDataloaderOutput
from acn_embed.util.base_trainer.base_inference_output import BaseInferenceOutput
from acn_embed.util.base_trainer.base_training_state import BaseTrainingState
from acn_embed.util.logger import get_logger
from acn_embed.util.torchutils import get_num_trainable_params

LOGGER = get_logger(__name__)


class BaseTrainer:

    # pylint: disable=too-many-instance-attributes
    def __init__(self, args):
        base_functions.dist_init()
        assert base_functions.dist_is_initialized()
        self.args = args
        if self.args.cpu or (not torch.cuda.is_available()):
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")
        self.model = None
        self.dataset_tr = None
        self.dataset_cv = None
        self.loader_tr = None
        self.loader_cv = None
        self.repeated_loader_tr = None
        self.optimizer = None
        self.wrapped_model = None
        self.scheduler = None
        self.ts = None

    def compute_cv_loss(self) -> float:
        self.wrapped_model.eval()
        tot_size = 0
        tot_summed_loss = 0
        with torch.no_grad():
            for cv_dataloader_out in self.loader_cv:
                cv_dataloader_out = cv_dataloader_out.to(device=self.device)

                # The loss is the mean over the input lengths and batch size.
                # We multiply by the batch size to get the "summed"
                inference_out = self.infer_1batch(dataloader_out=cv_dataloader_out)
                cv_loss_1batch, cv_size_1batch = self.compute_loss_1batch(
                    dataloader_out=cv_dataloader_out, inference_out=inference_out
                )
                tot_summed_loss += cv_loss_1batch.item()
                tot_size += cv_size_1batch.item()

        self.wrapped_model.train()

        tot_summed_loss, tot_size = base_functions.dist_get_summed(tot_summed_loss, tot_size)

        return float((tot_summed_loss / tot_size).detach().cpu())

    @abstractmethod
    def compute_loss_1batch(
        self, dataloader_out: BaseDataloaderOutput, inference_out: BaseInferenceOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss for 1 batch of data and return the summed loss and the batch size.
        The loss is not averaged over the batch, so a larger batch results in larger loss.
        The batch "size" can be in whatever units chosen by the implementation
        (number of sequences, or number of frames, etc.)
        """
        raise NotImplementedError()

    def end_of_run(self):
        # Usually a good idea to sync before ending
        base_functions.dist_barrier(timeout_seconds=3600)

    @abstractmethod
    def get_datasets(self):
        """
        Return dataset_tr, dataset_cv
        """
        raise NotImplementedError("")

    @abstractmethod
    def get_init_model(self) -> torch.nn.Module:
        raise NotImplementedError("")

    def get_last_lr(self):
        lr = self.scheduler.get_last_lr()
        if isinstance(lr, list):
            return lr[0]  # assume they're all the same
        return lr

    @abstractmethod
    def get_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError()

    @abstractmethod
    def get_scheduler(self):
        raise NotImplementedError()

    @abstractmethod
    def infer_1batch(self, dataloader_out: BaseDataloaderOutput) -> BaseInferenceOutput:
        raise NotImplementedError()

    def handle_min_cv_loss(self):
        """
        Do any handling for the occasion when the best CV loss has been observed
        """
        self.ts.best_epoch = self.ts.epoch
        self.ts.min_cv_loss = self.ts.cv_loss
        self.ts.best_epoch_tr_loss = self.ts.tr_loss
        self.ts.best_model_fn = self.ts.model_fn
        base_functions.log_local_leader(
            "info", f"Best cv-loss={self.ts.min_cv_loss:.3e} model={self.ts.best_model_fn}"
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

        if base_functions.is_local_leader():
            best_model_fn = os.path.join(self.args.model_output_local_dir, "model.best.pt")
            shutil.copyfile(self.ts.best_model_fn, best_model_fn)
            LOGGER.info(f"Updated {best_model_fn}")

    def train_1epoch(self, epoch) -> float:
        """
        Do backprop for 1 epoch and return the average loss over the training data.
        """

        def sample_params():
            """Get the first 3 params of the model, for debugging purposes"""
            for p in self.model.parameters():
                if p.requires_grad:
                    return torch.flatten(p).detach().cpu().numpy()[:3]
            return None

        tr_loss_sum = 0.0
        tr_size_sum = 0
        self.optimizer.zero_grad()
        for batch_idx in range(self.args.steps_per_epoch):
            tr_dataloader_output = next(self.repeated_loader_tr)
            tr_dataloader_output = tr_dataloader_output.to(device=self.device)

            inference_out = self.infer_1batch(dataloader_out=tr_dataloader_output)
            tr_loss_1batch, tr_size_1batch = self.compute_loss_1batch(
                dataloader_out=tr_dataloader_output, inference_out=inference_out
            )
            tr_loss_1batch_item = tr_loss_1batch.item()
            tr_size_1batch_item = tr_size_1batch.item()
            tr_loss_sum += tr_loss_1batch_item
            tr_size_sum += tr_size_1batch_item
            base_functions.log_local_leader("debug", "Doing backward()")
            tr_loss_1batch.backward()  # gradient gathering & model synchronization occurs here
            base_functions.log_local_leader("debug", "Completed backward()")
            self.optimizer.step()
            if batch_idx % self.args.log_steps == 0:
                base_functions.log_local_leader(
                    "info",
                    f"{epoch=} steps={self.scheduler.steps_completed} "
                    f"{batch_idx=} "
                    f"lr={self.get_last_lr():.3e} "
                    f"tr_loss_1batch={tr_loss_1batch_item:.3e} "
                    f"tr_size_1batch={tr_size_1batch_item} "
                    f"{self.wrapped_model.get_log_str()} "
                    f"sample_params={sample_params()} ",
                )
            del tr_loss_1batch  # this saves some gpu memory?
            self.scheduler.step()
            self.optimizer.zero_grad()

        tr_loss_sum, tr_size_sum = base_functions.dist_get_summed(tr_loss_sum, tr_size_sum)
        tr_loss_mean = tr_loss_sum / tr_size_sum
        return float(tr_loss_mean.detach().cpu())

    def get_new_training_state(self) -> BaseTrainingState:
        raise NotImplementedError()

    def load_checkpoint(self) -> BaseTrainingState:
        ts = self.get_new_training_state()
        filename = os.path.join(self.args.checkpoint_dir, "model.ckpt")
        if os.path.isfile(filename):
            map_location = {"cuda:0": f"cuda:{base_functions.dist_get_cuda_num()}"}
            checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
            ts = checkpoint["ts"]
            ts.epoch += 1  # Should increment 1 since we're going to start the next epoch
            self.model.load_state_dict(
                checkpoint["model_state_dict"],
            )
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            base_functions.log_local_leader(
                "info",
                f"Loaded checkpoint from {filename}. Starting training from epoch {ts.epoch}.",
            )
        else:
            base_functions.log_local_leader("info", "No checkpoint data found.")

        return ts

    def save_checkpoint(self, filename):
        if base_functions.can_save_checkpoint():
            base_functions.log_local_leader("info", f"Saving checkpoint at {filename}.")
            torch.save(
                {
                    "ts": self.ts,
                    "model_spec": self.wrapped_model.model_spec,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                filename,
            )
            base_functions.sync_checkpoint()
        base_functions.dist_barrier(timeout_seconds=3600)

    def save_model(self, epoch):
        model_fn = os.path.join(self.args.model_output_local_dir, f"model.{epoch:02d}.pt")
        if base_functions.is_local_leader():
            os.makedirs(self.args.model_output_local_dir, exist_ok=True)
            torch.save(
                {
                    "model_spec": self.wrapped_model.model_spec,
                    "model_state_dict": self.wrapped_model.state_dict(),
                },
                model_fn,
            )
            base_functions.log_local_leader("info", "Wrote %s", model_fn)
        if base_functions.is_leader() and self.args.model_output_leader_dir:
            leader_model_fn = os.path.join(
                self.args.model_output_leader_dir, f"model.{epoch:02d}.pt"
            )
            os.makedirs(self.args.model_output_leader_dir, exist_ok=True)
            torch.save(
                {
                    "model_spec": self.wrapped_model.model_spec,
                    "model_state_dict": self.wrapped_model.state_dict(),
                },
                leader_model_fn,
            )
            base_functions.log_leader("info", "Wrote %s", leader_model_fn)
        return model_fn

    def run(self):

        self.model = self.get_init_model()
        base_functions.log_local_leader(
            "info", f"Trainable model parameters {get_num_trainable_params(self.model):,d}"
        )

        self.dataset_tr, self.dataset_cv = self.get_datasets()

        self.loader_tr = torch.utils.data.DataLoader(
            batch_size=self.args.batch_size_per_node_tr,
            collate_fn=self.dataset_tr.collate,
            dataset=self.dataset_tr,
            drop_last=True,
            multiprocessing_context="spawn",
            num_workers=self.args.data_loader_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4 if base_functions.world_size() > 1 else None,
            shuffle=True,
        )
        self.loader_cv = torch.utils.data.DataLoader(
            batch_size=self.args.batch_size_per_node_cv,
            collate_fn=self.dataset_cv.collate,
            dataset=self.dataset_cv,
            drop_last=False,
            multiprocessing_context="spawn",
            num_workers=self.args.data_loader_workers,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4 if base_functions.world_size() > 1 else None,
            shuffle=False,
        )

        self.repeated_loader_tr = base_functions.infinite_dataloader(self.loader_tr)

        base_functions.log_local_leader(
            "info", f"{len(self.dataset_tr)=} " f"{len(self.dataset_cv)=}"
        )

        self.optimizer = self.get_optimizer()

        self.model, self.wrapped_model, self.optimizer = base_functions.get_dist_model(
            model=self.model, optimizer=self.optimizer
        )

        self.scheduler = self.get_scheduler()

        self.ts = self.load_checkpoint()
        self.model.train()

        while self.ts.epoch < self.args.max_epochs:
            epoch_start_time = time.time()
            self.ts.lrate = self.get_last_lr()
            self.ts.tr_loss = self.train_1epoch(epoch=self.ts.epoch)
            assert self.ts.tr_loss is not None
            self.ts.cv_loss = self.compute_cv_loss()
            assert self.ts.cv_loss is not None
            self.ts.epoch_time = time.time() - epoch_start_time
            base_functions.log_local_leader(
                "info", self.ts.log_str() + " " + self.wrapped_model.get_log_str()
            )

            if base_functions.is_leader():
                metrics_dict = self.ts.metrics()
                metrics_dict.update(self.wrapped_model.get_metrics())
                base_functions.log_leader_metrics(metrics_dict, self.ts.epoch)

            self.ts.model_fn = self.save_model(self.ts.epoch)

            if self.ts.min_cv_loss is None or self.ts.cv_loss < self.ts.min_cv_loss:
                self.handle_min_cv_loss()

            self.save_checkpoint(filename=os.path.join(self.args.checkpoint_dir, "model.ckpt"))
            if (
                base_functions.is_leader()
                and self.args.save_all_checkpoints
                and self.args.model_output_leader_dir
            ):
                self.save_checkpoint(
                    filename=os.path.join(
                        self.args.model_output_leader_dir, f"checkpoint.{self.ts.epoch:02d}.ckpt"
                    )
                )

            # Stop training if we've waited enough for the next best epoch
            if (self.args.wait_epochs is not None) and (
                self.ts.epoch >= self.ts.best_epoch + self.args.wait_epochs
            ):
                base_functions.log_local_leader(
                    "info", "Reached max number of wait_epochs. Stopping training."
                )
                break

            self.ts.epoch += 1

        self.end_of_run()
        base_functions.dist_shutdown()

    @staticmethod
    def add_args(parser):
        parser.add_argument("--batch-size-per-node-cv", action="store", type=int, required=True)
        parser.add_argument("--batch-size-per-node-tr", action="store", type=int, required=True)
        parser.add_argument(
            "--checkpoint-dir",
            action="store",
            type=str,
            required=True,
            help="Location to store the last trained model.",
        )
        parser.add_argument("--cpu", action="store_true", default=False)
        parser.add_argument("--data-loader-workers", action="store", type=int, default=2)
        parser.add_argument("--log-steps", action="store", type=int, required=True)
        parser.add_argument("--max-epochs", action="store", type=int, required=True)
        parser.add_argument(
            "--model-output-leader-dir",
            action="store",
            type=str,
            default=None,
            help="Model output location that only the global leader writes to. "
            "Usually a remote location.",
        )
        parser.add_argument(
            "--model-output-local-dir",
            action="store",
            type=str,
            default=".",
            help="A local model output location that only local leaders write to. "
            "Since the global leader is also a local leader, it would write to here too.",
        )
        parser.add_argument(
            "--save-all-checkpoints",
            action="store_true",
            default=False,
            help="Makes leader save a copy of all checkpoint files in --model-output-leader-dir",
        )
        parser.add_argument("--steps-per-epoch", action="store", type=int, required=True)
        parser.add_argument("--wait-epochs", action="store", type=int, default=None)
