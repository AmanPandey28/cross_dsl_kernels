from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedSpec:
    seed: int = 20260703


def seeded_torch_generator(seed: int = 20260703):
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen
