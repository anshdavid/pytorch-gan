import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.architecture import Discriminator, Generator, initialize_weights
from src.util import getDeveice


def main(args):

    device = getDeveice()

    # Hyperparameters etc.
    learningRate = args.lr
    epochs = args.epochs
    batchSize = args.batchsz

    imageSize = args.imagesz
    imageChannel = args.imagech
    dataFolder = args.datafl

    noiseDim = args.noisedim
    disFeature = args.disfea
    genFeature = args.genfea

    transforms_ = transforms.Compose([
        transforms.Resize(imageSize),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ),
    ])

    if args.log:
        writer_real = SummaryWriter(f"logs/real")
        writer_fake = SummaryWriter(f"logs/fake")
        writer = SummaryWriter(f"logs/")

    dataset = datasets.ImageFolder(
        root=dataFolder,
        transform=transforms_)


    dataloader = DataLoader(dataset, batch_size=batchSize)
    gen = Generator(noiseDim, imageChannel, genFeature).to(device)
    dis = Discriminator(imageChannel, disFeature).to(device)
    initialize_weights(gen)
    initialize_weights(dis)

    opt_gen = optim.Adam(gen.parameters(), lr=learningRate, betas=(0.5, 0.999))
    opt_dis = optim.Adam(dis.parameters(), lr=learningRate, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, noiseDim, 1, 1).to(device)
    step = 0

    gen.train()
    dis.train()

    for epoch in range(epochs):
        try:
            for batch_idx, (real, _) in enumerate(dataloader):
                real = real.to(device)
                noise = torch.randn(batchSize, noiseDim, 1, 1).to(device)
                fake = gen(noise)

                ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                dis_real = dis(real).reshape(-1)
                loss_dis_real = criterion(dis_real, torch.ones_like(dis_real))

                dis_fake = dis(fake.detach()).reshape(-1)
                loss_dis_fake = criterion(dis_fake, torch.zeros_like(dis_fake))

                loss_dis = (loss_dis_real + loss_dis_fake) / 2

                dis.zero_grad()
                loss_dis.backward()
                opt_dis.step()

                ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                output = dis(fake).reshape(-1)
                loss_gen = criterion(output, torch.ones_like(output))

                gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()

                # Print losses occasionally and print to tensorboard
                if batch_idx % 32 == 0:
                    with torch.no_grad():
                        if args.log:
                            fake = gen(fixed_noise)
                            img_grid_real = torchvision.utils.make_grid(
                                real[:32], normalize=True)
                            img_grid_fake = torchvision.utils.make_grid(
                                fake[:32], normalize=True)
                            writer_real.add_image("real", img_grid_real, global_step=step)
                            writer_fake.add_image("fake", img_grid_fake, global_step=step)

                step += 1
                if args.log:
                    writer.add_scalar('loss/train/discriminator', loss_dis, global_step=step)
                    writer.add_scalar('loss/train/generator', loss_gen, global_step=step)

                print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(dataloader)} \
                    Loss D: {loss_dis:.4f}, loss G: {loss_gen:.4f}")

            checkpoint = {
                "imageSize": imageSize,
                "generator": gen.state_dict(),
                "discriminator": dis.state_dict(),
                "genOptimizer": opt_gen.state_dict(),
                "disOptimizer": opt_dis.state_dict()}
            torch.save(checkpoint, "models/checkpoint.pt")

        except Exception as e:
            # print(e)
            pass