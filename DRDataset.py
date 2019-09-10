import os

from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DRDataset(Dataset):
    def __init__(self, csv_file, idx_list, dim, transformer):
        super(Dataset, self).__init__()
        # print('Dataset')
        self.idx_list = idx_list
        df = pd.read_csv(csv_file)
        self.data = df
        self.dim = dim
        self.transformer = transformer

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):
        img_name = os.path.join('data/new_data/train_images',
                                self.data.loc[self.idx_list[item], 'id_code'] + '.png')
        image = Image.open(img_name)
        image = image.resize((self.dim, self.dim), resample=Image.BILINEAR)
        label = torch.FloatTensor([self.data.loc[self.idx_list[item], 'diagnosis']])
        return self.transformer(image), label


class DRDatasetAlbumentation(Dataset):
    def __init__(self, csv_file, idx_list, dim, transformer):
        super(Dataset, self).__init__()
        self.idx_list = idx_list
        df = pd.read_csv(csv_file)
        self.data = df
        self.dim = dim
        self.transformer = transformer

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):
        img_name = os.path.join('data/new_data/train_images',
                                self.data.loc[self.idx_list[item], 'id_code'] + '.png')
        image = Image.open(img_name)
        image_np = np.float32(np.asarray(image))
        augmented = self.transformer(image=image_np)
        # Convert numpy array to PIL Image
        image = Image.fromarray(augmented['image'], mode='RGB')
        image = image.resize((self.dim, self.dim), resample=Image.BILINEAR)
        image_np = np.asarray(image)
        image = image_np.reshape(
            1, image_np.shape[0], image_np.shape[1], image_np.shape[2]).swapaxes(0, 3)
        image = image.reshape(image.shape[0], image.shape[1], image.shape[2])
        image = torch.FloatTensor(image)
        label = torch.FloatTensor([self.data.loc[self.idx_list[item], 'diagnosis']])
        return image, label

