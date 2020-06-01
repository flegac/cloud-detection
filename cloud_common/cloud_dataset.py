from enum import Enum

import cv2
import numpy as np
import pandas as pd

from cloud_common.dataset import Sample, Dataset

ROOT_PATH = '/DATADRIVE1/Datasets/38_cloud'


class CLoudBand(Enum):
    red = "red"
    green = "green"
    blue = "blue"
    nir = "nir"
    gt = "gt"

    def read(self, pattern: str, uri: str):
        return cv2.imread(pattern.format(band=self.value, uri=uri), cv2.IMREAD_GRAYSCALE)


class CloudSample(Sample):
    def __init__(self, pattern: str, uri: str):
        super().__init__(uri)
        self.pattern = pattern

    def x(self) -> np.ndarray:
        red = CLoudBand.red.read(self.pattern, self.uri)
        green = CLoudBand.green.read(self.pattern, self.uri)
        blue = CLoudBand.blue.read(self.pattern, self.uri)
        nir = CLoudBand.nir.read(self.pattern, self.uri)
        return np.dstack((red, green, blue, nir))

    def y(self) -> np.ndarray:
        return CLoudBand.gt.read(self.pattern, self.uri)


class CloudDataset(Dataset):
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id

    def items(self):
        root_path = '/DATADRIVE1/Datasets/38_cloud/38-Cloud_{dataset_id}'.format(dataset_id=self.dataset_id)
        pattern = root_path + '/{dataset_id}_{band}/{band}_{uri}.TIF'.replace('{dataset_id}', self.dataset_id)
        df = pd.read_csv(root_path + '/{dataset_id}_patches_38-Cloud.csv'.format(dataset_id=self.dataset_id))
        for uri in df['name']:
            yield CloudSample(pattern=pattern, uri=uri)


if __name__ == '__main__':
    train = CloudDataset('train')
    test = CloudDataset('test')

    for sample in train.items():
        assert sample.x().shape == (384, 384, 4)
        assert sample.y().shape == (384, 384)

    for sample in test.items():
        assert sample.x().shape == (384, 384, 4)
        assert sample.y().shape == (384, 384)
