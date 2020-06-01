import torch
from fastai.vision import models, unet_learner, F, DataBunch

from cloud_common.cloud_dataset import CloudDataset


def bce_logits_floatify(input, target, reduction='mean'):
    return F.binary_cross_entropy_with_logits(input, target.float(), reduction=reduction)


def dice_metric(pred, targs, threshold=0):
    pred = (pred > threshold).float()
    targs = targs.float()  # make sure target is float too
    return 2.0 * (pred * targs).sum() / ((pred + targs).sum() + 1.0)


if __name__ == '__main__':
    torch.cuda.empty_cache()

    data = CloudDataset('train')
    train, test = data.split(train_ratio=.8, test_ratio=.2)
    print(len(train), len(test))

    data = DataBunch.create(train, train, bs=2, num_workers=1)

    learn = unet_learner(
        data=data, arch=models.resnet18,
        loss_func=bce_logits_floatify,
        metrics=[dice_metric]
    )

    print(learn.model[0][0])
    print(learn.model[-1][-1])
    learn.fit(epochs=20)
