import numpy as np
from pandas import DataFrame
from data import AnimalDataset
from model import *
import matplotlib.pyplot as plt

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

device = "cuda" if torch.cuda.is_available() else "cpu"
predicate_binary_mat = np.array(np.genfromtxt('data/predicate-matrix-binary.txt', dtype='int'))
classes = np.array(np.genfromtxt('data/classes.txt', dtype='str'))[:, -1]
train_classes = np.array(np.genfromtxt('data/trainclasses.txt', dtype='str'))
test_classes = np.array(np.genfromtxt('data/testclasses.txt', dtype='str'))
predicate_continuous_mat = np.array(np.genfromtxt('data/predicate-matrix-continuous.txt', dtype='str'))
attribute = np.array(np.genfromtxt('data/predicates.txt', dtype='str'))[:, 1]

test2index={
    0:24,
    1:38,
    2:14,
    3:5,
    4:41,
    5:13,
    6:17,
    7:47,
    8:33,
    9:23
}
model = resnet50().to(device)
model.load_state_dict(torch.load("saved_model/24.pth"))
model.eval()

test_process_steps = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

test_loader = DataLoader(
    AnimalDataset('testclasses.txt', test_process_steps),
    batch_size=1,
    shuffle=False,
    num_workers=3,
)
pred_index = []
true_index = []
for i, (image, attribute, img_names, indexes) in enumerate(test_loader):
    images = image.to(device)
    atts = attribute.to(device).float()
    output = model(images)
    sigmoid_outputs = torch.sigmoid(output)
    ind = indexes.to(device)
    arr = np.array(sigmoid_outputs.detach().cpu().numpy()[0,:])
    score = []
    for j in [24,38,14,5,41,13,17,47,33,23]:
        pred = predicate_binary_mat[j,:]
        score.append(np.sqrt(np.sum((pred-arr)**2)))
    pred_index.append(test2index[score.index(min(score))])
    true_index.append(ind.item())
    print("{}/{}".format(i,len(test_loader)))

pred = np.array(pred_index)
true = np.array(true_index)
# df_pred = DataFrame(pred)
# df_true = DataFrame(true)
# df_pred.to_csv("pred.csv")
# df_true.to_csv("true.csv")
acc = np.mean(pred == true)
print(acc)