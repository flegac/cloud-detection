from enum import Enum

import cv2
import numpy as np
import pandas as pd
import torch
from fastai.vision import Image, SegmentationItemList

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
        data = np.dstack((red, green, blue))
        return np.moveaxis(data, -1, 0)

    def y(self) -> np.ndarray:
        data = CLoudBand.gt.read(self.pattern, self.uri)
        if data is None:
            return None
        data = np.dstack((data, 1 - data))
        data = np.moveaxis(data, -1, 0)
        return data.astype(np.float)


class CloudDataset(Dataset):
    def __init__(self, dataset_id: str):
        root_path = '/DATADRIVE1/Datasets/38_cloud/38-Cloud_{dataset_id}'.format(dataset_id=dataset_id)
        self.pattern = root_path + '/{dataset_id}_{band}/{band}_{uri}.TIF'.replace('{dataset_id}', dataset_id)
        self.df = pd.read_csv(root_path + '/{dataset_id}_patches_38-Cloud.csv'.format(dataset_id=dataset_id))
        self.dataset_id = dataset_id
        self.c = 2

    def __getitem__(self, index):
        uri = self.df['name'][index]
        sample = CloudSample(pattern=self.pattern, uri=uri)

        t1, t2 = torch.Tensor(sample.x()), torch.LongTensor(sample.y())

        return Image(t1), Image(t2)

    def __len__(self):
        return 5 # len(self.df)


class CloudImageItemList(SegmentationItemList):

    def open(self, uri: str):
        uri = uri.replace('./', '')
        dataset_id = 'train'
        root_path = '/DATADRIVE1/Datasets/38_cloud/38-Cloud_{dataset_id}'.format(dataset_id=dataset_id)
        pattern = root_path + '/{dataset_id}_{band}/{band}_{uri}.TIF'.replace('{dataset_id}', dataset_id)

        sample = CloudSample(pattern=pattern, uri=uri)
        return Image(torch.Tensor(sample.x()))
