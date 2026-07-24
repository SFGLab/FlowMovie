import tqdm
import torch
import numpy as np


def train_FM(model, optim, x0_npy, x1_npy, training_steps=500, batch_size=10):
    data_x0 = torch.from_numpy(x0_npy[:-1, :]).float()
    data_x1 = torch.from_numpy(x1_npy[:-1, :]).float()
    
    indices = torch.randperm(batch_size)
    
    # Split indices (80/20)
    train_size = int(0.8 * batch_size)
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    
    # Slice tensors
    x0_train, x1_train = data_x0[train_idx], data_x1[train_idx]
    x0_test, x1_test = data_x0[test_idx], data_x1[test_idx]
    
    pbar = tqdm.tqdm(range(training_steps))
    losses_train = []
    losses_test = []
    for i in pbar:
        # calculate train loss
        ids = torch.randint(x0_train.size(0), (batch_size,))
        x0 = x0_train[ids]
        x1 = x1_train[ids]
        target = x1 - x0
        t = torch.rand(x1.size(0))
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        pred = model(xt, t)
        loss_train = ((target - pred)**2).mean()
    
        # calculate test loss
        target = x1_test - x0_test
        t = torch.rand(x1_test.size(0))
        xt = (1 - t[:, None]) * x0_test + t[:, None] * x1_test
        pred = model(xt, t)
        loss_test = ((target - pred)**2).mean()
    
        # train
        loss_train.backward()
        optim.step()
        optim.zero_grad()
        pbar.set_postfix(loss=loss_train.item())
        losses_train.append(loss_train.item())
        losses_test.append(loss_test.item())

    return losses_train, losses_test


    