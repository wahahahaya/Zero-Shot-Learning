import numpy as np
from pandas import DataFrame
from data import AnimalDataset
from model import *
import loss

from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision
import torchvision.transforms as transforms

import os
import sys
import math


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Tensor = torch.cuda.FloatTensor if device == "cuda" else torch.Tensor

    # parameters
    epochs = 10
    last_epoch = 0
    sample_interval = 100

    # image transformations
    train_process = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.Resize((224,224)), # ImageNet standard
        transforms.ToTensor()
    ])

    # dataloader
    train_data = DataLoader(
        AnimalDataset('trainclasses.txt', train_process),
        batch_size=1,
        shuffle=True,
        num_workers=3,
    )

    # Build model
    gen = GenUnet().to(device)
    dis = Dis().to(device)


    # Loss Function
    adv_loss = torch.nn.BCELoss()
    pix_loss = torch.nn.MSELoss()
    fea_loss = torch.nn.MSELoss()
    ssim_loss = loss.SSIM()

    # optimizer
    opt_g = torch.optim.Adam(
        gen.parameters(),
        lr=0.0002,
        betas=(0.01, 0.999),
    )
    opt_d = torch.optim.Adam(
        dis.parameters(),
        lr=0.0002,
        betas=(0.01, 0.999),
    )
    def PSNR(img1, img2):
        MSE = torch.nn.MSELoss()
        psnr = 10*math.log10((255**2)/MSE(img1,img2).item())
        return psnr

    def sample_images(batches_done):
        imgs = next(iter(train_data))
        imgs = imgs[0].to(device)
        gen.eval()

        input = Variable(imgs.type(Tensor))
        output = gen(input)[0]


        ssim = loss.SSIM()
        ssim_loss = ssim(output,input).item()
        psnr_loss = PSNR(output,input)

        img_input = make_grid(input, nrow=5, normalize=True)
        img_output = make_grid(output, nrow=5, normalize=True)


        image_grid = torch.cat((img_input, img_output), 1)
        save_image(image_grid, "train_image/%s,SSIM=%.3f,PSNR=%.3f.png" % (batches_done, ssim_loss, psnr_loss), normalize=False)



    gg=[]
    for epoch in range(epochs):

        g=[]
        for i, (images, features, img_names, index) in enumerate(train_data):
            image = images.to(device)

            input = Variable(image.type(Tensor))
            # noise = Variable(image.type(Tensor))

            # Adversarial ground truths
            vaild = Variable(Tensor(np.ones((input.size(0), *dis.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((input.size(0), *dis.output_shape))), requires_grad=False)

            #
            # Train Generator
            #

            opt_g.zero_grad()

            output = gen(input)[0]
            g_loss = adv_loss(dis(output), vaild)

            ssim = ssim_loss(output, input)
            ssim_ = abs(ssim-1)
            pix = pix_loss(output, input)
            img_loss = ssim_ + pix



            total_loss = g_loss  + 0.5*pix + 1.5*ssim_
            g.append(total_loss.item())
            total_loss.backward()

            opt_g.step()

            #
            # Train Discriminator
            #

            opt_d.zero_grad()

            real_loss = adv_loss(dis(input), vaild)
            fake_loss = adv_loss(dis(output.detach()), fake)
            d_loss = real_loss + fake_loss

            d_loss.backward()
            opt_d.step()

            #
            # Log progress
            #
            batches_done = (epoch) * len(train_data) + i
            print("Epoch: {}/{}, Batch: {}/{}, D loss: {:.4f}, G loss: {:.4f}, img loss: {:.4f}, ssim: {:.4f}, pix: {:.4f}, total G: {:.4f}".format(epoch,last_epoch+epochs-1,i,len(train_data),d_loss.item(),g_loss.item(),img_loss.item(),ssim.item(),pix.item(),total_loss.item()))

            # If at sample interval save image
            if batches_done % sample_interval == 0:
                sample_images(batches_done)

        gg.append(sum(g) / len(g))
        torch.save(gen.state_dict(),"saved_model_unet/Gen_%d.pth" % (epoch))

    df_loss = DataFrame(gg)
    df_loss.to_csv("gan_loss.csv")


if __name__ == "__main__":
    train()