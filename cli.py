# -*- coding: utf-8 -*-

import argparse
import os.path

from trainer import main as train

def sanityCheck(args):

    train(args=args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='commands line arguments')

    parser.add_argument("--lr", type=int, default=2e-4, required=False, help="learning rate, deafault 2e-4")
    parser.add_argument( "--epochs", type=int, default=100, required=False, help="training epochs, default 100")
    parser.add_argument( "--batchsz", type=int, default=64, required=False, help="batch size, default 64")

    parser.add_argument( "--imagesz", type=int, default=64, required=False, help="image size, default 64")
    parser.add_argument( "--imagech", type=int, default=3, required=False, help="image channel, default 3")
    parser.add_argument( "--datafl", type=str, default='data/raw', required=False, help="data folder, default \'data/raw\'")

    parser.add_argument( "--noisedim", type=int, default=128, required=False, help="input noise dimension, default 128")
    parser.add_argument( "--disfea", type=int, default=64, required=False, help="discriminator features, default 64")
    parser.add_argument( "--genfea", type=int, default=64,  required=False, help="generator features, default 64")

    parser.add_argument ( '--log', action = 'store_false', help="logs, default true")

    args = parser.parse_args()

    sanityCheck(args=args)


