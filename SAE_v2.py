import scipy
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from data import AnimalDataset
from scipy import io, spatial, linalg
from pandas import DataFrame


device = "cuda" if torch.cuda.is_available() else "cpu"
def find_w(x, s, ld):
    A = np.dot(s, s.T)
    B = ld * np.dot(x, x.T)
    C = (1 + ld) * np.dot(s, x.T)
    w = scipy.linalg.solve_sylvester(A, B, C)
    return w

def img_feat(img):
    res101 = models.resnet101(pretrained=True)
    model = torch.nn.Sequential(*(list(res101.children())[:-1])).to(device)
    return model(img)

def normalizeFeature(x):
    x = x + 1e-10
    feat_norm = np.sum(x ** 2, axis=1)** 0.5
    feat = x / feat_norm[:, np.newaxis]
    return feat

def train_w():
    # image transformations
    train_process = transforms.Compose([
        transforms.Resize((224,224)), # ImageNet standard
        transforms.ToTensor()
    ])

    # dataloader
    data = DataLoader(
        AnimalDataset('trainclasses.txt', train_process),
        batch_size=10,
        shuffle=True,
        num_workers=3,
    )

    w = np.zeros((85,2048))
    for i, (images, features, img_names, index) in enumerate(data):
        images = images.to(device)
        attribute = features.detach().cpu().numpy()
        feature = img_feat(images)[:,:,0,0].detach().cpu().numpy().T
        w += find_w(normalizeFeature(feature), attribute.T, 500000)
        print("{}/{}".format(i + 1, len(data)))

    df_w = DataFrame(w)
    df_w.to_csv("SAE_W.csv")


if __name__ == "__main__":
    train_w()
