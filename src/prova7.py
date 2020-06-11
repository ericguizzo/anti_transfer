import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as utils
import define_models_torch as mod
import numpy as np
from torchvision import models
import configparser
import loadconfig


model = models.vgg16()

print (model)
