#!/usr/bin/env python3
import pickle
from pathlib import Path

import torch
import torch.distributed
import torch.multiprocessing
import torch.nn
import torch.utils.data
import yaml

import acn_embed.embed.model.g_embedder.utils
import acn_embed.embed.model.get_model
import acn_embed.util.torchutils
from acn_embed.embed.train.g.dataset import MinibatchDataset, MinibatchDatasetOutput
from acn_embed.embed.train.g.normalize import (
    get_input_meanvar_normalizer_dist,
    get_output_meanvar_normalizer_dist,
)
from acn_embed.util.base_trainer import base_functions
from acn_embed.util.base_trainer.base_inference_output import BaseInferenceOutput
from acn_embed.util.base_trainer.base_trainer import BaseTrainer
from acn_embed.util.base_trainer.base_training_state import BaseTrainingState
from acn_embed.util.logger import get_logger
from acn_embed.util.three_phase_scheduler import ThreePhaseScheduler
from acn_embed.util.torchutils import get_num_trainable_params

LOGGER = get_logger(__name__)


# pylint: disable=duplicate-code


class InferenceOutput(BaseInferenceOutput):
    def __init__(self, output_t: torch.Tensor):
        super().__init__()
        self.output_t = output_t


class GTrainer(BaseTrainer):
    def __init__(self, args, model_spec, scheduler_spec):
        self.subword_idx_to_str, self.subword_str_to_idx = (
            acn_embed.embed.model.g_embedder.utils.get_subword_info(
                subword_json_path=args.subword_json
            )
        )

        super().__init__(args)

        self.model_spec = model_spec
        self.scheduler_spec = scheduler_spec

        base_functions.log_local_leader(
            "info", "model_spec=\n" + yaml.safe_dump(model_spec, sort_keys=False)
        )
        base_functions.log_local_leader(
            "info", "scheduler_spec=\n" + yaml.safe_dump(scheduler_spec, sort_keys=False)
        )

        rank = base_functions.get_rank()
        self.foutput_path_tr = self.args.data_path_tr / f"foutput.{rank}.pkl"
        self.segment_path_tr = self.args.data_path_tr / f"g-segments.{rank}.pkl.gz"
        self.foutput_path_cv = self.args.data_path_cv / f"foutput.{rank}.pkl"
        self.segment_path_cv = self.args.data_path_cv / f"g-segments.{rank}.pkl.gz"

    def get_datasets(self):
        dataset_tr = MinibatchDataset(
            foutput_path=self.foutput_path_tr,
            segment_path=self.segment_path_tr,
            subword_type=self.args.subword,
            subword_str_to_id=self.subword_str_to_idx,
        )
        dataset_cv = MinibatchDataset(
            foutput_path=self.foutput_path_cv,
            segment_path=self.segment_path_cv,
            subword_type=self.args.subword,
            subword_str_to_id=self.subword_str_to_idx,
        )
        return dataset_tr, dataset_cv

    def get_init_model(self) -> torch.nn.Module:
        with self.foutput_path_tr.open("rb") as fobj:
            dim_out = pickle.load(fobj)[-1]

        self.model_spec["class_args"]["dim_in"] = len(self.subword_idx_to_str)
        self.model_spec["class_args"]["dim_out"] = dim_out
        self.model_spec["class_args"]["idx_to_subword"] = self.subword_idx_to_str

        model = acn_embed.embed.model.get_model.get(model_spec=self.model_spec).to(self.device)
        base_functions.log_local_leader(
            "info", f"Trainable model parameters {get_num_trainable_params(model):,d}"
        )

        base_functions.log_local_leader("info", "Normalizing inputs...")
        model.in_normalizer = get_input_meanvar_normalizer_dist(
            segment_path=self.segment_path_tr,
            subword_type=self.args.subword,
            subword_str_to_id=self.subword_str_to_idx,
        )
        base_functions.log_local_leader("info", "Normalizing outputs...")
        model.out_normalizer = get_output_meanvar_normalizer_dist(
            foutput_path=self.foutput_path_tr
        )
        model = model.to(self.device)
        return model

    def get_new_training_state(self) -> BaseTrainingState:
        return BaseTrainingState()

    def get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters())

    def get_scheduler(self):
        return ThreePhaseScheduler(optimizer=self.optimizer, **self.scheduler_spec)

    def infer_1batch(self, dataloader_out: MinibatchDatasetOutput) -> InferenceOutput:
        output_t = self.model(dataloader_out)
        if torch.isnan(output_t).any():
            raise RuntimeError("NaN found in output. Aborting.")
        return InferenceOutput(output_t=output_t)

    def compute_loss_1batch(
        self, dataloader_out: MinibatchDatasetOutput, inference_out: InferenceOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.sum((dataloader_out.ref_t - inference_out.output_t) ** 2)
            / self.wrapped_model.dim_out,
            torch.tensor(inference_out.output_t.shape[0]),
        )

    @staticmethod
    def add_args(parser):
        BaseTrainer.add_args(parser)
        parser.add_argument("--data-path-cv", action="store", type=Path, required=True)
        parser.add_argument("--data-path-tr", action="store", type=Path, required=True)
        parser.add_argument("--decay1-steps-k", action="store", type=float, default=None)
        parser.add_argument("--decay2-factor-k", action="store", type=float, default=None)
        parser.add_argument("--dropout-prob", action="store", type=float, default=0.4)
        parser.add_argument("--dim-state", action="store", type=int, default=None, required=False)
        parser.add_argument("--max-lrate", action="store", type=float, required=True)
        parser.add_argument("--num-layers", action="store", type=int, default=None, required=False)
        parser.add_argument(
            "--subword", action="store", choices=["grapheme", "phone"], required=True
        )
        parser.add_argument("--subword-json", action="store", type=Path, required=True)
        parser.add_argument("--warmup-steps-k", action="store", type=float, default=None)
