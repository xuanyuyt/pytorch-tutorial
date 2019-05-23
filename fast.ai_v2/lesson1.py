# -*- coding=utf-8 -*-

# import torch
# import torchvision
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #
# 1. Load and visualize the data            (Line 22 to 87)
# 2. Define a neural network                (Line 89 to 125)
# 3. Train the model                        (Line 127 to 186)
# 4. Evaluate the performance of our
# trained model on a test dataset!           (Line 188 to 252)


from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

PATH = "G:/Other_Datasets/dogscats/"
sz=224
# ================================================================== #
# 1.加载预训练模型，模型默认保存在 C:\Users\MyPC\.torch\models 路径下
# from torchvision.models import resnet34 (.torch_imports)
arch=resnet34
# ================================================================== #
# 2.加载数据
# tfms_from_model (.transforms)
# ImageClassifierData (.dataset)
# data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
# ================ augmentation
# tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
# def get_augs():
#     data = ImageClassifierData.from_paths(PATH, bs=2, tfms=tfms, num_workers=1)
#     x,_ = next(iter(data.aug_dl))
#     return data.trn_ds.denorm(x)[0]
# ims = np.stack([get_augs() for i in range(6)])
# plots(ims, rows=2)
# plt.tight_layout()
# plt.show()
# ================ augmentation
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms)
# ================================================================== #
# 3.创建网络
# learn = ConvLearner.pretrained(arch, data, precompute=True)
learn = ConvLearner.pretrained(arch, data, precompute=False)
# ================================================================== #
# 4.训练
# ================ learning rate
# lrf=learn.lr_find()
# learn.sched.plot_lr()
# learn.sched.plot()
# ================ learning rate
learn.unfreeze()
lr=np.array([1e-4,1e-3,1e-2])
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.sched.plot_lr()
# ================================================================== #
# 5.保存模型
learn.save('224_tmp') # path/models
learn.load('224_tmp')
log_preds,y = learn.TTA()
probs = np.mean(np.exp(log_preds),0)
accuracy(probs, y)


