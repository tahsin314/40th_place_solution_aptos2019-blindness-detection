# Courtesy: https://medium.com/udacity-pytorch-challengers/ideas-on-how-to-fine-tune-a-pre-trained-model-in-pytorch-184c47185a20

import math


def cyclical_lr(stepsize, min_lr=3e-2, max_lr=3e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


def triangular_lr(iteration, cycle_length, min_lr, max_lr, decay_rate=0.1):
    iteration -= 1  # if iteration count starts at 1
    cycle = math.floor(1 + iteration / cycle_length)
    x = abs(iteration / (cycle_length / 2) - 2 * cycle + 1)
    new_lr = min_lr + (max_lr - min_lr) * max(0, (1 - x))*math.exp(-decay_rate*iteration/cycle_length)
    return new_lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr