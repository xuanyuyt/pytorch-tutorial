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

PATH = "G:/Other_Datasets/dogscats/"
sz=224
arch=resnet34 # from torchvision.models import resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(0.01, 2)


