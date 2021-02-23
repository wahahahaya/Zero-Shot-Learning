import numpy as np
from pandas import DataFrame
from data import AnimalDataset
from model import resnet50

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision
import torchvision.transforms as transforms

import os
import sys

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # parameters
    num_epochs = 100
    last_epoch = 0

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
        batch_size=24,
        shuffle=True,
        num_workers=1,
    )

    # Build model
    model = resnet50().to(device)

    # loss function
    criterion = torch.nn.BCELoss()

    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.01, 0.999))

    total_steps = len(train_data)
    batch_loss=[]
    for epoch in range(num_epochs):
        for i, (images, features, img_names, index) in enumerate(train_data):
            if images.shape[0] < 2:
                break
            images = images.to(device)
            features = features.to(device).float()

            model.train()

            outputs = model(images)
            sigmoid_outputs = torch.sigmoid(outputs)
            loss = criterion(sigmoid_outputs, features)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Batch Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_steps, loss.item()))
                batch_loss.append(loss.item())
                sys.stdout.flush()

        torch.save(model.state_dict(), "saved_model/%d.pth" % (epoch))

    df_loss = DataFrame(batch_loss)
    df_loss.to_csv("loss.csv")

if __name__ == "__main__":
    train()
