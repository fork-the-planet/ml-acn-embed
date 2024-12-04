#!/usr/bin/env python3
import json
import time
from enum import Enum
from pathlib import Path

import numpy as np
import torch

import acn_embed.embed.model.get_model
import acn_embed.util.torchutils
from acn_embed.util.base_inference_input import BaseInferenceInput
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class GraphemeOrPhone(Enum):
    GRAPHEME = "grapheme"
    PHONE = "phone"


class TextEmbedder:

    def __init__(self, model_dir: Path, text_type: str, device=None):
        """
        Initialize a text embedder for computing Acoustic Neighbor Embeddings

        model_dir:
            Directory containing model files

        text_type:
            "phone" or "grapheme"

        device:
            The device to run inference on. Will auto-detect if not specified.
        """
        assert model_dir.is_dir()
        self.model_dir = model_dir

        if device:
            self.device = device
        else:
            self.device = torch.device("cpu")
            if torch.cuda.is_available():
                self.device = torch.device("cuda")

        assert text_type in ["grapheme", "phone"], "text_type must be either grapheme- or phone"
        self.grapheme_or_phone = GraphemeOrPhone[text_type.upper()]

        self.text_type = text_type
        self.model_dir = model_dir

        embedder_fn = model_dir / f"text-{text_type}.pt"
        from_file = torch.load(embedder_fn, map_location=self.device, weights_only=True)
        self.model_spec = from_file["model_spec"]
        self.model = acn_embed.embed.model.get_model.get(model_spec=self.model_spec).to(
            self.device
        )
        self.model.load_state_dict(from_file["model_state_dict"])
        self.model.eval()

        cluster_fn = model_dir / f"{text_type}-cluster-stats.json"
        if cluster_fn.exists():
            with cluster_fn.open("r", encoding="utf8") as fobj:
                self.sigma = json.load(fobj)["mean_of_clusterwise_std"]

    def _preprocess_text(self, text):
        """
        Preprocess text before feeding into nnet
        """
        # pylint: disable=use-a-generator
        if self.grapheme_or_phone == GraphemeOrPhone.PHONE:
            assert all(
                [isinstance(elem, list) for elem in text]
            ), "For phone embedder, all elements of 'text' must be a list"
        elif self.grapheme_or_phone == GraphemeOrPhone.GRAPHEME:
            assert all(
                [isinstance(elem, str) for elem in text]
            ), "For grapheme embedder, all elements of 'text' must be a str"

        for _text in text:
            for sw in _text:
                if sw not in self.model.subword_to_idx:
                    LOGGER.warning(f"Dropping illegal subword {sw}")

        if self.grapheme_or_phone == GraphemeOrPhone.GRAPHEME:
            filtered_text = [
                "".join([ch for ch in _text if ch in self.model.subword_to_idx]) for _text in text
            ]
        else:
            filtered_text = [
                [ph for ph in _text if ph in self.model.subword_to_idx] for _text in text
            ]
        return filtered_text

    def _get_padded_input(self, text):

        num_frames = np.array([len(inp) for inp in text])
        if np.any(num_frames == 0):
            raise RuntimeError("Encountered empty subword sequence after sanitization")

        # pylint: disable=not-callable
        sequences = [
            torch.nn.functional.one_hot(
                torch.LongTensor(subword_id_list), num_classes=self.model.num_subwords
            ).to(dtype=torch.float32)
            for subword_id_list in text
        ]

        padded_input_t, len_t = acn_embed.util.torchutils.pad_sequence_batch_first(
            sequences=sequences, min_len=self.model.min_input_len
        )

        return padded_input_t, len_t

    def get_embedding(self, text: list, batch_size: int = 500, log_interval: int = 100):
        """
        Get Acoustic Neighbor Embeddings for a list of text (subword sequences)

        text:
            A list of iterables, each iterable being a sequence of subwords
            e.g. For a phone model, [["P","AE1", "R", "IH0", "S"],["F", "R", "AE1", "N", "S"]]
                 For a grapheme model, ["THE CITY OF PARIS", "FRANCE"].
                 A space (" ") is just like another character.

        batch_size:
            The size of each batch during inference. Use if the `text` list is large.

        log_interval:
            The interval for logging progress.

        Returns:
            A 2-d PyTorch tensor where each row stores an embedding for one sequence.
            The number of rows should be the same as the number of sequences.
        """

        def log():
            nonlocal start_idx, text, start_time
            if len(text) <= 1:
                return
            percent = 100 if start_idx == len(text) else int((100.0 * start_idx) / len(text))
            sequences_per_sec = start_idx / (time.time() - start_time)
            LOGGER.info(
                f"Processing sequence {start_idx:,d} / {len(text):,d} "
                f"({percent:d} %) {sequences_per_sec:.2f} strings/sec"
            )

        text = self._preprocess_text(text)

        with torch.no_grad():
            start_idx = 0
            arrs = []
            batch_num = 0
            start_time = time.time()
            while start_idx < len(text):
                if log_interval and batch_num % log_interval == 0:
                    log()
                end_idx = min(start_idx + batch_size, len(text))
                _text = [
                    [
                        self.model.subword_to_idx[subword]
                        for subword in text[idx]
                        if subword in self.model.subword_to_idx
                    ]
                    for idx in range(start_idx, end_idx)
                ]

                padded_input_t, len_t = self._get_padded_input(_text)

                LOGGER.debug("padded_input_t.shape=%s", padded_input_t.shape)
                LOGGER.debug("len_t=%s", len_t)

                goutput = self.model.forward(
                    BaseInferenceInput(
                        input_t=padded_input_t.to(device=self.device),
                        input_len_t=len_t.to(device=torch.device("cpu")),
                    )
                )
                arrs.append(goutput.detach().cpu())
                start_idx = end_idx
                batch_num += 1
            log()
            return torch.cat(arrs, dim=0)
