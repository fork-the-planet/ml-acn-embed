#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.special
import torch

from acn_embed.embed.embedder.text_embedder import TextEmbedder
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


class ExpectedConfusion:
    def __init__(
        self,
        *,
        str2score: Path,
        embeddings: Path,
        grapheme_embedder_path: Path,
        lm_scales: list,
        sigma: float,
    ):

        with open(str2score, "rb") as fobj:
            obj = torch.load(fobj, weights_only=True, map_location="cpu")
            self.strings = obj["strings"]
            self.lmscores = np.log(10) * np.array(obj["scores"])

        with open(embeddings, "rb") as fobj:
            self.embeddings = (
                torch.load(fobj, weights_only=True, map_location="cpu").detach().numpy()
            )

        assert len(self.strings) == self.lmscores.size == self.embeddings.shape[0]

        self.embedder = TextEmbedder(model_dir=grapheme_embedder_path, text_type="grapheme")

        self.lm_scales = np.array([lm_scales])  # [1, num_lm_scales]
        assert self.lm_scales.shape == (1, len(lm_scales))
        self.sigma = sigma

    def _get_data(self, wakeup_word):
        filtered_strings = []
        filtered_emb = []
        filtered_scores = []
        for idx, string in enumerate(self.strings):
            if wakeup_word in string:
                continue
            # Hack for "okay"
            if wakeup_word.replace("OKAY", "OK") in string:
                continue
            filtered_emb.append(np.expand_dims(self.embeddings[idx, :], axis=0))
            filtered_strings.append(string)
            filtered_scores.append(self.lmscores[idx])
        return filtered_strings, np.concatenate(filtered_emb, axis=0), filtered_scores

    def compute(self, wakeup_word):
        strings, embeddings, lmscores = self._get_data(wakeup_word)

        wakeup_embedding = self.embedder.get_embedding(text=[wakeup_word]).detach().cpu().numpy()
        ac_score = np.expand_dims(
            np.sum(np.power(embeddings - wakeup_embedding, 2.0), axis=1)
            / (8 * self.sigma * self.sigma)
            + np.log(2),
            -1,
        )  # shape = (vocab_size, 1)
        lm_score = np.expand_dims(np.array(lmscores), -1)  # shape = (vocab_size, 1)
        score = np.dot(ac_score, self.lm_scales - 1) + np.dot(
            lm_score, self.lm_scales
        )  # (vocab_size, num_lm_scales)

        # Print sum of the nearest neighbors
        for lm_idx in range(score.shape[1]):
            LOGGER.info(f"lm_scale = {self.lm_scales[0, lm_idx]:.2f}")
            sorted_idx = np.argpartition(-score[:, lm_idx], kth=10)
            for _idx in sorted_idx[:10]:
                LOGGER.info(f"{strings[_idx]} {score[_idx, lm_idx]:4f}")

        score = scipy.special.logsumexp(score, axis=0, keepdims=True)  # (1, num_lm_scales)
        return score


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Get embeddings and LM scores",
    )
    parser.add_argument("--embeddings", action="store", type=Path, required=True)
    parser.add_argument("--str2score", action="store", type=Path, required=True)
    parser.add_argument("--model", action="store", type=Path, required=True)
    parser.add_argument("--sigma", action="store", type=float, required=True)
    parser.add_argument("--output", action="store", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    lm_scales = np.arange(0.80, 1.01, 0.01).tolist()

    ec = ExpectedConfusion(
        str2score=args.str2score,
        embeddings=args.embeddings,
        grapheme_embedder_path=args.model,
        lm_scales=lm_scales,
        sigma=args.sigma,
    )

    wakeup_words = ["KITT", "MOTHER", "THREEPIO", "COMPUTER", "TEE ONE THOUSAND"]

    conf_list = []
    for wakeup_word in wakeup_words:
        LOGGER.info(f"*** {wakeup_word=} ***")
        conf = ec.compute(wakeup_word)
        LOGGER.info(f"{conf=}")
        conf_list.append(conf)
    confs = np.concatenate(conf_list, axis=0)  # (num_wakeup_words, num_lm_scales)

    with args.output.open("wb") as fobj:
        torch.save(
            {"wakeup_words": wakeup_words, "confs": confs.tolist(), "lm_scales": lm_scales}, fobj
        )


if __name__ == "__main__":
    main()
