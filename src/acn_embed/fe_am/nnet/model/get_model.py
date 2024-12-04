from acn_embed.fe_am.nnet.model.conformer_am import ConformerAm


def get(model_spec: dict):
    if model_spec["class"] == "ConformerAm":
        model = ConformerAm(**model_spec["class_args"])
    else:
        raise RuntimeError(f"Unrecognized type {model_spec['class']}")
    model.model_spec = model_spec
    return model
