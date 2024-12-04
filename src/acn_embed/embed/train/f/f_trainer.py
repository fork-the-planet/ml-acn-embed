#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import torch
import torch.distributed
import torch.multiprocessing
import torch.nn
import torch.utils.data
import yaml

import acn_embed.embed.model.get_model
from acn_embed.embed.train.f.normalize import get_input_meanvar_normalizer_dist
from acn_embed.embed.train.f.smart_microbatch_dataset import (
    SmartMicroBatchDataset,
    SmartMicroBatchDatasetOutput,
)
from acn_embed.fe_am.nnet.train.training_dataset import TrainingDatasetOutput
from acn_embed.util.base_trainer import base_functions
from acn_embed.util.base_trainer.base_inference_output import BaseInferenceOutput
from acn_embed.util.base_trainer.base_trainer import BaseTrainer
from acn_embed.util.base_trainer.base_training_state import BaseTrainingState
from acn_embed.util.logger import get_logger
from acn_embed.util.three_phase_scheduler import ThreePhaseScheduler
from acn_embed.util.torchutils import get_num_trainable_params

LOGGER = get_logger(__name__)


class InferenceOutput(BaseInferenceOutput):
    def __init__(self, output_t: torch.Tensor):
        super().__init__()
        self.output_t = output_t


class FTrainer(BaseTrainer):
    def __init__(self, args, model_spec, scheduler_spec):
        super().__init__(args)

        self.model_spec = model_spec
        self.scheduler_spec = scheduler_spec

        base_functions.log_local_leader(
            "info", "model_spec=\n" + yaml.safe_dump(model_spec, sort_keys=False)
        )
        base_functions.log_local_leader(
            "info", "scheduler_spec=\n" + yaml.safe_dump(scheduler_spec, sort_keys=False)
        )

    def get_datasets(self):
        dataset_tr = SmartMicroBatchDataset(
            data_path=self.args.data_path_tr,
            len_microbatch=self.args.mbatch_size,
            max_same=self.args.mbatch_max_same,
            min_input_len=self.model.min_input_len,
            num_phones_to_prior=self.args.num_phones_to_prior,
            tot_num_pivots=self.args.pivots_k_tr * 1000,
        )
        dataset_cv = SmartMicroBatchDataset(
            data_path=self.args.data_path_cv,
            len_microbatch=self.args.mbatch_size,
            max_same=self.args.mbatch_max_same,
            min_input_len=self.model.min_input_len,
            num_phones_to_prior=self.args.num_phones_to_prior,
            tot_num_pivots=self.args.pivots_k_cv * 1000,
            divide_by=base_functions.world_size(),
        )
        return dataset_tr, dataset_cv

    def get_init_model(self) -> torch.nn.Module:
        model = acn_embed.embed.model.get_model.get(model_spec=self.model_spec).to(self.device)
        base_functions.log_local_leader(
            "info", f"Trainable model parameters {get_num_trainable_params(model):,d}"
        )
        base_functions.log_local_leader("info", "Normalizing inputs...")
        model.in_normalizer = get_input_meanvar_normalizer_dist(
            h5_path=(self.args.data_path_tr / "amfeat.h5"),
            num_splits=base_functions.world_size(),
            split=base_functions.get_rank() + 1,
        )
        model = model.to(self.device)
        return model

    def get_new_training_state(self) -> BaseTrainingState:
        return BaseTrainingState()

    def get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters())

    def get_scheduler(self):
        return ThreePhaseScheduler(optimizer=self.optimizer, **self.scheduler_spec)

    def infer_1batch(self, dataloader_out: TrainingDatasetOutput) -> InferenceOutput:
        output_t = self.model(dataloader_out)
        if torch.isnan(output_t).any():
            raise RuntimeError("NaN found in output. Aborting.")
        return InferenceOutput(output_t=output_t)

    def compute_loss_1batch(
        self, dataloader_out: SmartMicroBatchDatasetOutput, inference_out: InferenceOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:

        log_fdiff_hyp_list = []
        foutput_idx = 0
        num_sequences = dataloader_out.input_len_t.numel()
        assert num_sequences == inference_out.output_t.shape[0]
        while foutput_idx < num_sequences:
            log_foutput_diff = -torch.sum(
                torch.pow(
                    inference_out.output_t[
                        foutput_idx + 1 : foutput_idx + dataloader_out.microbatch_size, :
                    ]
                    - inference_out.output_t[foutput_idx, :],
                    2.0,
                ),
                dim=1,
            )
            log_foutput_diff = log_foutput_diff - torch.logsumexp(log_foutput_diff, dim=0)
            foutput_idx += dataloader_out.microbatch_size
            log_fdiff_hyp_list.append(log_foutput_diff)

        tot_f_dist = None
        for _log_fdiff_hyp_t, _fdiff_ref_t, _num_same in zip(
            log_fdiff_hyp_list, dataloader_out.fdiff_ref_t, dataloader_out.num_same, strict=True
        ):
            _tot_f_dist = -np.log(_num_same) - torch.sum(_fdiff_ref_t * _log_fdiff_hyp_t)
            if tot_f_dist is None:
                tot_f_dist = _tot_f_dist
            else:
                tot_f_dist += _tot_f_dist

        return tot_f_dist, torch.tensor(len(log_fdiff_hyp_list))

    @staticmethod
    def add_args(parser):
        BaseTrainer.add_args(parser)
        parser.add_argument("--data-path-cv", action="store", type=Path, required=True)
        parser.add_argument("--data-path-tr", action="store", type=Path, required=True)
        parser.add_argument("--decay1-steps-k", action="store", type=float, required=True)
        parser.add_argument("--decay2-factor-k", action="store", type=float, required=True)
        parser.add_argument("--dim-out", action="store", type=int, required=True)
        parser.add_argument("--dim-state", action="store", type=int, required=True)
        parser.add_argument("--dropout-prob", action="store", type=float, required=True)
        parser.add_argument("--max-lrate", action="store", type=float, required=True)
        parser.add_argument("--mbatch-max-same", type=int, required=True)
        parser.add_argument("--mbatch-size", type=int, required=True)
        parser.add_argument("--num-layers", action="store", type=int, required=True)
        parser.add_argument("--num-phones-to-prior", action="store", type=str, required=True)
        parser.add_argument("--pivots-k-tr", action="store", type=int, required=True)
        parser.add_argument("--pivots-k-cv", action="store", type=int, required=True)
        parser.add_argument("--warmup-steps-k", action="store", type=float, required=True)
