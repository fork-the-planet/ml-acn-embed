from pathlib import Path

import torch

from acn_embed.embed.model.f_embedder.f_lstm_embedder import FLstmEmbedder
from acn_embed.embed.model.g_embedder.g_lstm_embedder import GLstmEmbedder


def get(model_spec: dict):
    if model_spec["class"] == "FLstmEmbedder":
        model = FLstmEmbedder(**model_spec["class_args"])
    elif model_spec["class"] == "GLstmEmbedder":
        model = GLstmEmbedder(**model_spec["class_args"])
    else:
        raise RuntimeError(f"Unrecognized type {model_spec['class']}")
    model.model_spec = model_spec
    return model


def load(file: Path, device=torch.device("cpu")):
    from_file = torch.load(file, map_location=device, weights_only=True)
    embedder = get(model_spec=from_file["model_spec"]).to(device)
    embedder.load_state_dict(from_file["model_state_dict"])
    embedder.eval()
    return embedder
