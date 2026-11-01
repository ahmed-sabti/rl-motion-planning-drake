import hashlib
from dataclasses import dataclass

import numpy as np

def _label_key(label: str) -> int:
    return int.from_bytes(hashlib.blake2b(label.encode("utf-8"), digest_size=4).digest(), "big")

@dataclass(frozen=True, slots=True)
class SeedSpec:
    entropy: int

    def stream(self, *labels: str) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence(
            self.entropy, spawn_key=tuple(_label_key(label) for label in labels)))
