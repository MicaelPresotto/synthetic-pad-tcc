from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

from synthetic_pad.domain.image import RgbImage


class AttackGenerator(ABC):
    name: ClassVar[str]

    @abstractmethod
    def generate(self, image: RgbImage) -> RgbImage: ...

    @abstractmethod
    def describe(self) -> Mapping[str, object]: ...


class LivenessDetector(ABC):
    name: ClassVar[str]

    @abstractmethod
    def score(self, image: RgbImage) -> float:
        """A score in [0, 1]. Higher means more likely to be an attack."""
        ...
