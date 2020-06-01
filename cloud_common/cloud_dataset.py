from enum import Enum
from random import shuffle
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from fastai.vision import Image, SegmentationItemList

from cloud_common.dataset import Sample, Dataset

ROOT_PATH = 'D:\\Datasets/38_cloud'


class CLoudBand(Enum):
    red = "red"
    green = "green"
    blue = "blue"
    nir = "nir"
    gt = "gt"

    def read(self, pattern: str, uri: str):
        data = cv2.imread(pattern.format(band=self.value, uri=uri), cv2.IMREAD_GRAYSCALE)
        data = (1 + data - data.min()) / (1 + data.max() - data.min())
        return data


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
        # data = np.dstack((data, 1 - data))
        # data = np.moveaxis(data, -1, 0)
        data = np.expand_dims(data, 0)
        return data.astype(np.float)


class CloudDataset(Dataset):
    def __init__(self, dataset_id: str, c: int = 1, indices: List[int] = None):
        root_path = ROOT_PATH + '/38-Cloud_{dataset_id}'.format(dataset_id=dataset_id)
        self.pattern = root_path + '/{dataset_id}_{band}/{band}_{uri}.TIF'.replace('{dataset_id}', dataset_id)
        self.df = pd.read_csv(root_path + '/{dataset_id}_patches_38-Cloud.csv'.format(dataset_id=dataset_id))
        self.dataset_id = dataset_id
        self.c = c
        self.indices = indices

    def __getitem__(self, index):
        if self.indices:
            index = self.indices[index]
        uri = self.df['name'][index]
        sample = CloudSample(pattern=self.pattern, uri=uri)

        t1 = torch.tensor(sample.x(), dtype=torch.float32)
        t2 = torch.tensor(sample.y(), dtype=torch.int64)
        return Image(t1), Image(t2)

    def __len__(self):
        if self.indices:
            return len(self.indices)
        return len(self.df)

    def split(self, train_ratio: float = .8, test_ratio: float = .2) -> Tuple['CloudDataset', 'CloudDataset']:
        total = len(self)
        test_size = int(total * test_ratio)
        train_size = int(total * train_ratio)
        indices = list(range(len(self)))
        shuffle(indices)
        train, test = indices[:train_size].copy(), indices[train_size:train_size + test_size].copy()
        return (
            CloudDataset(self.dataset_id, self.c, train),
            CloudDataset(self.dataset_id, self.c, test)
        )


class CloudImageItemList(SegmentationItemList):

    def open(self, uri: str):
        uri = uri.replace('./', '')
        dataset_id = 'train'
        root_path = ROOT_PATH + '/38-Cloud_{dataset_id}'.format(dataset_id=dataset_id)
        pattern = root_path + '/{dataset_id}_{band}/{band}_{uri}.TIF'.replace('{dataset_id}', dataset_id)

        sample = CloudSample(pattern=pattern, uri=uri)
        return Image(torch.tensor(sample.x()))
