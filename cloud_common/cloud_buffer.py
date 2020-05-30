from enum import Enum

import cv2
import numpy as np
import pandas as pd

from cloud_common.buffer import Buffer


class CLoudBand(Enum):
    red = "red"
    green = "green"
    blue = "blue"
    nir = "nir"
    gt = "gt"

    def read(self, uri: str):
        return cv2.imread('D://Datasets/clouds/train_{band}_additional_to38cloud/{band}_{uri}.TIF'.format(
            band=self.value,
            uri=uri
        ), cv2.IMREAD_GRAYSCALE)


class CloudBuffer(Buffer):
    def data(self) -> np.ndarray:
        red = CLoudBand.red.read(self.uri)
        green = CLoudBand.green.read(self.uri)
        blue = CLoudBand.blue.read(self.uri)
        nir = CLoudBand.nir.read(self.uri)
        return np.dstack((red, green, blue, nir))

    def gt(self) -> np.ndarray:
        return CLoudBand.gt.read(self.uri)


if __name__ == '__main__':
    df = pd.read_csv('D://Datasets/clouds/training_patches_95-cloud_nonempty.csv')
    for uri in df['name']:
        buffer = CloudBuffer(uri=uri)
        data = buffer.data()
        try:
            print(data.shape, data.min(), data.max())
        except:
            print('error on : {}'.format(uri))
