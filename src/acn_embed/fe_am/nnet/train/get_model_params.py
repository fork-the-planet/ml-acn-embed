def get_conformer(size):
    assert size in ["10M", "20M", "30M", "50M"]
    params = None
    if size == "10M":
        params = {"embedder_dim": 164, "num_heads": 4, "num_layers": 16, "conv_kernel_size": 31}
    elif size == "20M":
        params = {"embedder_dim": 232, "num_heads": 4, "num_layers": 16, "conv_kernel_size": 31}
    elif size == "30M":
        params = {"embedder_dim": 272, "num_heads": 8, "num_layers": 18, "conv_kernel_size": 31}
    elif size == "50M":
        params = {"embedder_dim": 300, "num_heads": 10, "num_layers": 23, "conv_kernel_size": 31}

    return params
