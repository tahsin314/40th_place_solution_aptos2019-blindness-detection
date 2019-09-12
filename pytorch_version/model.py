import torch
import torchvision
from torch import nn


def DRModel(device):
    model = torchvision.models.resnet101(pretrained=True)
    # model.load_state_dict(torch.load("../input/pytorch_version-pretrained-models/resnet101-5d3b4d8f.pth"))
    num_features = model.fc.in_features
    model.fc = nn.Linear(2048, 1)
    model = model.to(device)
    return model
