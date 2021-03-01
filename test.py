import numpy as np
from pandas import DataFrame
from data import AnimalDataset
from model import *

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
def labels_to_class(pred_labels):
    pred = []
    for i in range(pred_labels.shape[0]):
        curr_labels = pred_labels[i,:].cpu().detach().numpy()
        best_dist = sys.maxsize
        best_index = -1
        for j in range(predicate_binary_mat.shape[0]):
            class_labels = predicate_binary_mat[j,:]
            dist = np.sqrt(np.sum((curr_labels - class_labels)** 2))
            if dist < best_dist and classes[j] not in train_classes:
                best_index = j
                best_dist = dist
        pred.append(classes[best_index])
    return pred


def eval(model, dataloader):
    mean_acc = 0.0
    pred_classes = []
    true_classes = []
    for i, (image, features, img_names, indexes) in enumerate(dataloader):
        images = image.to(device)
        features = features.to(device).float()
        output = model(images)
        sigmoid_outputs = torch.sigmoid(output)
        pred_labels = sigmoid_outputs
        curr_pred_classes = labels_to_class(pred_labels)
        pred_classes.extend(curr_pred_classes)

        curr_true_classes = []
        for index in indexes:
            curr_true_classes.append(classes[index])
        true_classes.extend(curr_true_classes)

    pred_classes = np.array(pred_classes)
    true_classes = np.array(true_classes)
    mean_acc = np.mean(pred_classes == true_classes)

    return mean_acc, pred_classes, true_classes

def test():
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
        shuffle=True,
        num_workers=3,
    )

    curr_acc, pred_classes, true_classes = eval(model, test_loader)
    print(curr_acc)
    df_pred = DataFrame(pred_classes)
    df_true = DataFrame(true_classes)

    df_pred.to_csv("pred.csv")
    df_true.to_csv("true.csv")


if __name__ == "__main__":
    test()