# Courtesy: https://www.kaggle.com/tanlikesmath/intro-aptos-diabetic-retinopathy-eda-starter
import torch
from sklearn.metrics import cohen_kappa_score


def quadratic_kappa(y_hat, y, device: str = 'cuda:0'):
    return torch.tensor(cohen_kappa_score(torch.round(y_hat), y, weights='quadratic'), device=device)