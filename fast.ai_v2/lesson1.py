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


# ===========================load dataset ========================== #
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

PATH = "E:/Other_Datasets/dogscats/"
sz=224
# ======================== 1.加载预训练模型 ======================== #
# 1.加载预训练模型
# from torchvision.models import resnet34 (.torch_imports)
arch=resnet34
# =========================== 2.加载数据 =========================== #
# 2.加载数据
# tfms_from_model (.transforms)
# ImageClassifierData (.dataset)
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(arch, data, precompute=True)
# ============================= 3.训练 ============================= #
learn.fit(0.01, 2)


