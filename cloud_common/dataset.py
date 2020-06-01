from abc import ABC, abstractmethod

import numpy as np
import torch.utils.data

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


class Dataset(torch.utils.data.Dataset):
    glossary = Glossary(name='Dataset', definition='Describes a set of {sample}s'.format(
        sample=Sample.glossary.name
    ))

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
