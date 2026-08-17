from abc import ABC, abstractmethod
import numpy as np


class LivenessDetector(ABC):
    """Scores how likely an image is a presentation attack."""

    name: str

    @abstractmethod
    def score(self, image: np.ndarray) -> float:
        """Returns a score in [0, 1]. Higher means more likely to be an attack."""
        ...