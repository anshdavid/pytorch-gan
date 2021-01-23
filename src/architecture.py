# -*- coding: utf-8 -*-

from typing import Generator
import torch

import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU

class Generator(nn.Module):

    def __init__(self, in_noise, channels_img, features_g):

        super(Generator, self).__init__()

        self.gen = nn.Sequential(

            # img: 4x4
            nn.ConvTranspose2d(
                in_noise,
                features_g * 16,
                4,
                1,
                0,
                bias=False,
            ),
            nn.BatchNorm2d(features_g * 16),
            nn.ReLU(),

            # img: 8x8
            nn.ConvTranspose2d(
                features_g * 16,
                features_g * 8,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(),

            # img: 16x16
            nn.ConvTranspose2d(
                features_g * 8,
                features_g * 4,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(),

            # img: 32x32
            nn.ConvTranspose2d(
                features_g * 4,
                features_g * 2,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(),

            # img: 64x64
            nn.ConvTranspose2d(
                features_g * 2,
                channels_img,
                4,
                2,
                1,
                bias=False
            ),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):

    def __init__(self, channels_img, features_d):

        super(Discriminator, self).__init__()

        self.dis = nn.Sequential(

            # img: 64 x 64
            nn.Conv2d(
                channels_img,
                features_d,
                4,
                2,
                1
            ),
            nn.LeakyReLU(0.2),

            # img: 32 x 32
            nn.Conv2d(
                features_d,
                features_d * 2,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2),

            # img: 16 x 16
            nn.Conv2d(
                features_d * 2,
                features_d * 4,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2),

            # img: 8 x 8
            nn.Conv2d(
                features_d * 4,
                features_d * 8,
                4,
                2,
                1,
                bias=False
            ),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2),

            # img: 8 x 8
            nn.Conv2d(
                features_d * 8,
                1,
                4,
                2,
                0,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dis(x)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (
            nn.Conv2d,
            nn.ConvTranspose2d,
            nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)