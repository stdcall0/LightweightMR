import numpy as np
import torch
import torch.nn as nn

try:
    import tinycudann as tcnn
except ImportError as err:  # pragma: no cover - dependency hint
    raise ImportError(
        "tiny-cuda-nn is required for HashEncoder. "
        "Install it via `pip install -e ./tiny-cuda-nn` or add it to PYTHONPATH."
    ) from err


class HashEncoder(nn.Module):
    """tiny-cuda-nn HashGrid wrapper matching the previous interface."""

    def __init__(
        self,
        input_dim=3,
        num_levels=16,
        level_dim=2,
        per_level_scale=2,
        base_resolution=16,
        log2_hashmap_size=19,
        desired_resolution=None,
    ):
        super().__init__()

        if desired_resolution is not None and num_levels > 1:
            per_level_scale = np.exp2(
                np.log2(desired_resolution / base_resolution) / (num_levels - 1)
            )

        self.input_dim = int(input_dim)
        self.num_levels = int(num_levels)
        self.level_dim = int(level_dim)
        self.per_level_scale = float(per_level_scale)
        self.base_resolution = int(base_resolution)
        self.log2_hashmap_size = int(log2_hashmap_size)
        self.output_dim = self.num_levels * self.level_dim

        encoding_config = {
            "otype": "HashGrid",
            "n_levels": self.num_levels,
            "n_features_per_level": self.level_dim,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": self.base_resolution,
            "per_level_scale": self.per_level_scale,
        }
        self.encoding = tcnn.Encoding(
            n_input_dims=self.input_dim,
            encoding_config=encoding_config,
        )

    def __repr__(self):
        return (
            "HashEncoder(tcnn): input_dim={} num_levels={} level_dim={} "
            "base_resolution={} per_level_scale={:.4f}".format(
                self.input_dim,
                self.num_levels,
                self.level_dim,
                self.base_resolution,
                self.per_level_scale,
            )
        )

    def forward(self, inputs, size=1):
        size = float(size)
        normalized = (inputs + size) / (2 * size)
        prefix_shape = list(normalized.shape[:-1])
        flat_inputs = normalized.reshape(-1, self.input_dim)
        if not flat_inputs.is_cuda:
            raise ValueError("HashEncoder expects CUDA tensors; ensure inputs are on GPU.")
        if flat_inputs.dtype != torch.float32:
            flat_inputs = flat_inputs.float()

        outputs = self.encoding(flat_inputs)
        outputs = outputs.view(prefix_shape + [self.output_dim])
        return outputs