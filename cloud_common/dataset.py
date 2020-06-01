from abc import ABC, abstractmethod
from typing import Generator

import numpy as np

from cloud_common.glossary import Glossary


class Sample(ABC):
    glossary = Glossary(name='Sample', definition='Describes a single pair (input, expected) as numpy arrays.')
    uri: str

    def __init__(self, uri: str):
        self.uri = uri

    @abstractmethod
    def x(self) -> np.ndarray:
        ...

    @abstractmethod
    def y(self) -> np.ndarray:
        ...


class Dataset(ABC):
    glossary = Glossary(name='Dataset', definition='Describes a set of {sample}s'.format(
        sample=Sample.glossary.name
    ))

    @abstractmethod
    def items(self) -> Generator[Sample, None, None]:
        ...
