# -*- coding=utf-8 -*-

from fastai.imports import *
from fastai.torch_imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

PATH = "G:/Other_Datasets/dogbreed/"
sz = 224
arch = resnext101_64
bs = 10
label_csv = f'{PATH}labels.csv'
n = len(list(open(label_csv))) - 1 # header is not counted (-1)
val_idxs = get_cv_idxs(n) # random 20% data for validation set
# ================================================================== #
# label_df = pd.read_csv(label_csv)
# print(label_df.head())
# print(label_df.pivot_table(index="breed", aggfunc=len).sort_values('id', ascending=False))
# ================================================================== #
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
# ================================================================== #
# data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv', test_name='test', # we need to specify where the test set is if you want to submit to Kaggle competitions
#                                    val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)
# fn = os.path.join(data.trn_ds.path + data.trn_ds.fnames[0])
# print(fn)
# size_d = {k: PIL.Image.open(PATH + k).size for k in data.trn_ds.fnames}
# row_sz, col_sz = list(zip(*size_d.values()))
# row_sz = np.array(row_sz); col_sz = np.array(col_sz)
# plt.hist(row_sz)
# plt.show()
# plt.cla()
# plt.hist(row_sz[row_sz < 1000])
# plt.show()
# plt.cla()
# plt.hist(col_sz)
# plt.show()
# plt.cla()
# plt.hist(col_sz[col_sz < 1000])
# plt.show()
# print(len(data.trn_ds), len(data.test_ds))
# print(len(data.classes), data.classes[:5])
# ================================================================== #
# def get_data(sz, bs):  # sz: image size, bs: batch size
#     tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
#     data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv', test_name='test',
#                                         val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)
#     # Reading the jpgs and resizing is slow for big images, so resizing them all to 340 first saves time
#     return data if sz > 300 else data.resize(340, 'tmp')
# data = get_data(sz, bs)
# ================================================================== #
PATH = "G:/Other_Datasets/dogbreed/tmp/340/"
data = ImageClassifierData.from_csv(PATH, 'train', f'{PATH}labels.csv', test_name='test', # we need to specify where the test set is if you want to submit to Kaggle competitions
                                   val_idxs=val_idxs, suffix='.jpg', tfms=tfms, bs=bs)
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(1e-2, 5)