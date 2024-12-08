#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from acn_embed.embed.embedder.text_embedder import TextEmbedder
from acn_embed.util.logger import get_logger

LOGGER = get_logger(__name__)


def _prune_by_lm_score(strings, lmscores, embeddings, lm_score_thres):
    use_idx = np.nonzero(lmscores > lm_score_thres)[0]
    strings = [strings[idx] for idx in use_idx]
    embeddings = embeddings[use_idx, :]
    return strings, lmscores, embeddings


class NNSearch:
    def __init__(
        self,
        *,
        device: torch.device,
        embeddings_path: Path,
        grapheme_embedder_path: Path,
        lm_score_thres: float,
        strings_path: Path,
    ):

        self.device = device

        print("Reading strings...")
        with open(strings_path, "rb") as fobj:
            obj = torch.load(fobj, weights_only=True, map_location="cpu")
            self.strings = obj["strings"]
            self.lmscores = np.log(10) * np.array(obj["scores"])

        print("Reading embeddings...")
        with open(embeddings_path, "rb") as fobj:
            self.embeddings = torch.load(fobj, weights_only=True, map_location=device).detach()

        self.strings, self.lmscores, self.embeddings = _prune_by_lm_score(
            self.strings, self.lmscores, self.embeddings, lm_score_thres
        )

        print(f"Loaded {len(self.strings)} strings after pruning")

        self.embedder = TextEmbedder(
            model_dir=grapheme_embedder_path, text_type="grapheme", device=device
        )

    def search(self, query_word, num_requested):

        query_embedding = (
            self.embedder.get_embedding(text=[query_word]).detach().to(device=self.device)
        )
        l2_dist = torch.sqrt(torch.sum(torch.pow(self.embeddings - query_embedding, 2.0), dim=1))
        values, indices = torch.topk(
            l2_dist, dim=0, k=num_requested * 2, largest=False, sorted=True
        )
        shown = rank = 0
        while shown < num_requested and rank < len(values):
            dist = values[rank]
            result_string = self.strings[indices[rank]]
            if result_string != query_word:
                print(f"{dist=:.3f} {result_string}")
                shown += 1
            rank += 1


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Interactive nearest-neighbor search for phonetically-similar words",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--embeddings", action="store", type=Path, required=True)
    parser.add_argument("--lm-score-thres", type=float, default=-14)
    parser.add_argument("--model", action="store", type=Path, required=True)
    parser.add_argument("--num-results", type=int, default=5)
    parser.add_argument("--strings", action="store", type=Path, required=True)
    LOGGER.info(" ".join(sys.argv))
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    nnsearch = NNSearch(
        device=device,
        embeddings_path=args.embeddings,
        grapheme_embedder_path=args.model,
        lm_score_thres=args.lm_score_thres,
        strings_path=args.strings,
    )

    while True:
        query = input("\nQUERY>> ")
        if not query:
            return
        nnsearch.search(query.upper(), num_requested=args.num_results)


if __name__ == "__main__":
    main()
