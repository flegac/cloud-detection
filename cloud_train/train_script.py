import pandas as pd
from fastai.vision import models, unet_learner, DataBunch, F

from cloud_common.cloud_dataset import CloudDataset


def bce_logits_floatify(input, target, reduction='mean'):
    return F.binary_cross_entropy_with_logits(input, target.float(), reduction=reduction)


def dice_metric(pred, targs, threshold=0):
    pred = (pred > threshold).float()
    targs = targs.float()  # make sure target is float too
    return 2.0 * (pred * targs).sum() / ((pred + targs).sum() + 1.0)


if __name__ == '__main__':
    dataset_id = 'train'
    root_path = '/DATADRIVE1/Datasets/38_cloud/38-Cloud_{dataset_id}'.format(dataset_id=dataset_id)
    csv = '/{dataset_id}_patches_38-Cloud.csv'.format(dataset_id=dataset_id)
    df = pd.read_csv(root_path + csv)


    def label_func(x):
        return root_path + '/train_gt/gt_{}.TIF'.format(x.replace('./', ''))


    # imageList = CloudImageItemList.from_df(df, '.', cols='name')
    # data = (imageList
    #         .split_by_rand_pct()
    #         .label_from_func(label_func, classes=['none', 'cloud'])
    #         .databunch(bs=2, num_workers=2))

    train = CloudDataset('train')
    test = CloudDataset('test')
    data = DataBunch.create(train, train, bs=1, num_workers=1)

    learn = unet_learner(
        data=data, arch=models.resnet18,
        loss_func=bce_logits_floatify,
        metrics=[dice_metric])

    # learn.lr_find()
    # learn.recorder.plot()

    print(learn.model[0][0])
    print(learn.model[-1][-1])
    learn.fit(epochs=5)
