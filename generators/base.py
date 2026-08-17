from abc import ABC, abstractmethod
import numpy as np


class AttackGenerator(ABC):
    """Turns a bona fide face image into a synthetic presentation attack."""

    name: str

    @abstractmethod
    def generate(self, image: np.ndarray) -> np.ndarray:
        """Takes an RGB image (H, W, 3) uint8 and returns an attack sample."""
        ...