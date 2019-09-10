import math
from matplotlib import pyplot as plt
from torchvision import transforms

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt

x = [i for i in range(10)]
folds = KFold(n_splits=5, shuffle=True, random_state = 45)
for i in range(3):
    for i, j in folds.split(x):
        print(i,j)
    print('\n\n')
from DRDataset import DRDataset
transform = transforms.Compose([
    # transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
x = DRDataset('data/train.csv', [i for i in range(10)], 32, transform) 
print(iter(x)[0])