# PYTORCH GAN

## Usage

- to run training `cli.py`
    - optional arguments:
    - -h, --help           show this help message and exit
    - --lr LR              learning rate, deafault 2e-4
    - --epochs EPOCHS      training epochs, default 100
    - --batchsz BATCHSZ    batch size, default 64
    - --imagesz IMAGESZ    image size, default 64
    - --imagech IMAGECH    image channel, default 3
    - --datafl DATAFL      data folder, default 'data/raw'
    - --noisedim NOISEDIM  input noise dimension, default 128
    - --disfea DISFEA      discriminator features, default 64
    - --genfea GENFEA      generator features, default 64
    - --log                logs, type_action: default true

- to generate fake samples: ...
## Report

After `epochs: 20` and with `batch-size: 64`

+ Loss Discriminator

![Loss_Discriminator](reports/figures/loss-discriminator.svg?style=center "normal xray activation map")

+ Loss Generator

![Loss_Generator](reports/figures/loss-generator.svg?style=center "normal xray activation map")

+ Fake

![Fake_Sample](reports/figures/sample-fake.png?style=center "normal xray activation map")

+ Real

![Real_Sample](reports/figures/sample-real.png?style=center "normal xray activation map")

<!-- <p align="center">
    <img width="250" height="300" src="resources/logo.png">
</p> -->