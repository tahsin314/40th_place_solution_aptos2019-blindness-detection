import sys

import torch
from torch import optim, nn
from torchvision import transforms
from DRDataset import DRDataset
from model import DRModel
from cyclic_lr import get_lr, triangular_lr, set_lr
from kappa import quadratic_kappa
from logger import logger
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm as T
import platform

num_worker = None
if platform.system() == 'Windows':
    num_worker = 0
else:
    num_worker = 4

device = torch.device("cuda:0")
batch_size = 48
dim = 256
num_fold = 5
epochs = 10
step_size = 4*int(len(os.listdir('../data/new_data/train_images'))*(1.-1/num_fold)/batch_size)
model_dir = 'resnet101_mse_dim_64'
model = DRModel(device)

transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
# transform = Compose([
#         # Resize(width=256, height=256),
#         # transforms.CenterCrop(224),
#         Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
#                     HorizontalFlip(),
#                     OneOf([Rotate(limit=10),
#                     RandomBrightnessContrast(),
#                     RandomScale(scale_limit=0.2)],
#                           )])
plist = [
        {'params': model.layer4.parameters(), 'lr': 1e-5, 'weight': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-4}
    ]


optimizer = optim.Adam(plist, lr=1e-4)

criterion = nn.MSELoss()


def train(transformer, epoch, num_fold):
    try:
        os.mkdir('models/'+model_dir)
    except:
        pass

    kf = KFold(n_splits=num_fold, shuffle=True, random_state=42)
    df = pd.read_csv('../data/new_data/train.csv')
    for cv_num, (train_list, val_list) in enumerate(kf.split(df)):
        best_qk = 0
        best_loss = np.inf
        for e in T(range(epoch)):
            train_dataset = DRDataset('../data/new_data/train.csv', train_list, dim, transformer)
            train_data_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
            model.train()
            running_loss = 0.0
            with T(total=len(train_data_loader), file=sys.stdout) as pbar:
                for count, (data, labels) in T(enumerate(train_data_loader), total=len(train_data_loader)):
                    labels = labels.view(-1, 1)
                    data = data.to(device, dtype=torch.float)
                    labels = labels.to(device, dtype=torch.float)
                    optimizer.zero_grad()
                    it_num = cv_num*epochs*len(train_data_loader) + e*len(train_data_loader) + count + 1

                    # Verify this formula
                    lr = triangular_lr(it_num, 4*len(train_data_loader)*num_fold, 3e-5, 3e-4, 0.15)
                    set_lr(optimizer, lr)

                    with torch.set_grad_enabled(True):
                        outputs = model(data)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item() * data.size(0)
                    epoch_loss = running_loss / len(train_data_loader.dataset)
                    pbar.set_description('Loss: {:.4f}'.format(running_loss / ((count + 1) * batch_size)))
                    pbar.update(1)
                print('\nTraining Loss: {:.4f}'.format(epoch_loss))
                val_qk, val_loss = eval(val_list, transformer)
                logger(cv_num+1, e+1, get_lr(optimizer), epoch_loss, val_loss,
                       val_qk.data.cpu().numpy(), 'resnet101_dim_256_logger.csv')
                if val_qk > best_qk and val_loss < best_loss:
                    print(' -----------------------------')
                    print('|       New best model!       |')
                    print(' -----------------------------')
                    best_qk = val_qk
                    best_loss = val_loss
                    torch.save({
                    'epoch': e,
                    'cv_num':cv_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss,
                    'kappa': best_qk
                    }, 'models/'+model_dir+'/'+model_dir+'_fold_'+str(cv_num)+'.pth')


def eval(val_list, transformer):
    running_loss = 0.0
    predictions = []
    actual_labels = []
    val_dataset = DRDataset('../data/new_data/train.csv', val_list, dim, transformer)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size//2, shuffle=True, num_workers=num_worker)
    model.eval()
    for data, labels in T(val_data_loader):
        labels = labels.view(-1, 1)
        labels = labels.to(device, dtype=torch.float)
        data = data.to(device, dtype=torch.float)
        preds = model(data)
        loss = criterion(preds, labels).data.cpu().numpy()
        running_loss += loss.item() * data.size(0)
        predictions.extend(preds.data.cpu().numpy())
        actual_labels.extend(labels.data.cpu())
    epoch_loss = running_loss / len(val_data_loader.dataset)
    print('Validation Loss: {:.4f}'.format(epoch_loss))
    qk = quadratic_kappa(torch.Tensor(predictions), torch.Tensor(actual_labels)) 
    print('Quadratic Kappa: {:.4f}'.format(qk))
    return qk, epoch_loss


kf = KFold(n_splits=num_fold)
df = pd.read_csv('../data/new_data/train.csv')
train(transform,
    epochs, 5)
