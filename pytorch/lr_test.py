import torch
from sklearn.model_selection import KFold
from torch import optim, nn

import pandas as pd
from torchvision import transforms

from pytorch.lr_finder import LRFinder
from pytorch.model import DRModel
from pytorch.DRDataset import DRDataset
device = torch.device("cuda:0")
model = DRModel(device)
df = pd.read_csv('data/train.csv')
kf = KFold(n_splits=4)
batch_size = 10
dim = 256
transform = transforms.Compose([
    transforms.Resize(256),
    # transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)

for train_df, val_df in kf.split(df):
    dataset = DRDataset('data/train.csv', train_df, dim, transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(data_loader, end_lr=100, num_iter=300)
    lr_finder.plot()
    break
