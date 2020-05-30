from abc import ABC, abstractmethod

import numpy as np

from cloud_common.glossary import Glossary


class Buffer(ABC):
    glossary = Glossary(name='Buffer', definition='Describes how to access some numpy array (data/buffer).')
    uri: str

    def __init__(self, uri: str):
        self.uri = uri

    @abstractmethod
    def data(self) -> np.ndarray:
        ...