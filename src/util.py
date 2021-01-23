# -*- coding: utf-8 -*-

import torch.nn as nn
from torch import cuda as cuda_

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
def getDeveice():
    if cuda_.is_available():
        return "cuda:0"
    else:
        return "cpu"