import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class resnet50(torch.nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()
        model = models.resnet50()
        fc_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.BatchNorm1d(fc_features),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(fc_features, 85)
        )
        self.resnet_fea = model

    def forward(self, x):
        x = self.resnet_fea(x)

        return(x)