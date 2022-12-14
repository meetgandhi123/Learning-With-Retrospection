# -*- coding: utf-8 -*-
"""Learning with Retrospection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1j_Di4z5p6fh_f2uL0QnaiKs_gzakwhTx
"""

import numpy as np
import torch
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
import torch.nn as nn


def soft_crossentropy(logits, y_true, dim):
    return -1 * (torch.log_softmax(logits, dim=dim) * y_true).sum(axis=1).mean(axis=0)


def crossentropy(logits, y_true, dim):
    if dim == 1:
        return F.cross_entropy(logits, y_true)
    else:
        loss = 0.0
        for i in range(logits.shape[1]):
            loss += soft_crossentropy(logits[:, i, :], y_true[:, i, :], dim=1)
        return loss


class LWR(torch.nn.Module):
    def __init__(
        self,
        k: int,
        num_batches_per_epoch: int,
        dataset_length: int,
        output_shape: Tuple[int],
        max_epochs: int,
        tau=5.0,
        update_rate=0.9,
        softmax_dim=1,
        use_kl=False,
    ):
        """
        Args:
            k: int, Number of Epochs after which soft labels are updated (interval)
            num_batches
        """
        super().__init__()
        self.k = k
        self.update_rate = update_rate
        self.max_epochs = max_epochs

        self.step_count = 0
        self.epoch_count = 0
        self.num_batches_per_epoch = num_batches_per_epoch

        self.tau = tau
        self.alpha = 1.0
        self.scaling = 4

        self.softmax_dim = softmax_dim

        self.labels = torch.zeros((dataset_length, *output_shape))
        self.usekl = use_kl

    def forward(
        self,
        batch_idx: Tensor,
        logits: Tensor,
        y_true: Tensor,
        previous_output=None,
        eval=False,
    ):
        self.alpha = 1 - self.update_rate * self.epoch_count * self.k / self.max_epochs
        if self.epoch_count <= self.k:
            self.step_count += 1
            if (
                self.step_count + 1
            ) % self.num_batches_per_epoch == 0 and eval is False:
                self.step_count = 0
                self.epoch_count += 1

            if self.epoch_count == self.k and eval is False:
                # print(self.labels[batch_idx, ...].shape, logits.shape)
                if self.usekl:
                    self.labels[batch_idx, ...] = (
                        torch.softmax(logits / self.tau, dim=self.softmax_dim)
                        .detach()
                        .clone()
                        .cpu()
                    )
            return F.cross_entropy(logits, y_true)
        else:
            if (self.epoch_count + 1) % self.k == 0 and eval is False and use_kl:
                if self.usekl:
                    self.labels[batch_idx, ...] = (
                        torch.softmax(logits / self.tau, dim=self.softmax_dim)
                        .detach()
                        .clone()
                        .cpu()
                    )
            if self.usekl:
                return self.loss_fn_with_kl(logits, y_true, batch_idx)
            else:
                return self.L1_loss_fn(logits, y_true, previous_output)

    def loss_fn_with_kl(
        self, logits: Tensor, y_true: Tensor, batch_idx: Tensor,
    ):
        # assert(logits.shape == y_true.shape)
        return self.alpha * crossentropy(logits, y_true, dim=self.softmax_dim) + (
            1 - self.alpha
        ) * self.tau * self.tau * F.kl_div(
            F.log_softmax(logits / self.tau, dim=self.softmax_dim),
            self.labels[batch_idx, ...].to(logits.get_device()),
            reduction="batchmean",
        )

    def L1_loss_fn(self, logits: Tensor, y_true: Tensor, previous_output: Tensor):
        """
        From Jandial, Surgan, et al. 
        "Retrospective Loss: Looking Back 
        to Improve Training of Deep Neural Networks." 
        Proceedings of the 26th ACM SIGKDD 
        International Conference on Knowledge 
        Discovery & Data Mining. 2020.
        """
        task_loss = F.cross_entropy(logits, y_true)
        logits = F.softmax(logits, dim=1)
        previous_output = F.softmax(previous_output, dim=1)


        b = nn.L1Loss()(logits, previous_output.detach())
        a = nn.L1Loss()(logits, y_true)

        retrospective_loss = ((self.scaling + 1) * a) - ((self.scaling) * b)

        return task_loss + retrospective_loss

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

global_step = 0

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def train(usekl,model, device, train_loader, optimizer, epoch, lwr, k, snapshot=None):
    model.train()
    for i, (batch_idx, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if usekl:
            output = model(data)
            loss = lwr(batch_idx, output, target, eval=False)

        else:
            output = model(data)
            if epoch >= k:
                assert snapshot != None
                previous_output = snapshot(data)
            else:
                previous_output = None

            loss = lwr(batch_idx, output, target, previous_output, eval=False)

        loss.backward()
        optimizer.step()

        global global_step
        #writer.add_scalar("train/loss", loss.item(), global_step)
        global_step += 1
        if i % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    i * len(data),
                    len(train_loader.dataset),
                    100.0 * i / len(train_loader),
                    loss.item(),
                )
            )
        #     if args.dry_run:
        #         break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    global global_step
    #writer.add_scalar("test/loss", test_loss, global_step)
    #writer.add_scalar("test/acc", correct / len(test_loader.dataset), global_step)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return idx, self.ds[idx][0], self.ds[idx][1]

use_cuda = True
torch.manual_seed(1)
batch_size=32
test_batch_size=32
lr=0.0001
gamma=0.7
epochs=20
k=5
usekl=True

device = torch.device("cuda" if use_cuda else "cpu")
train_kwargs = {"batch_size": batch_size}
test_kwargs = {"batch_size": test_batch_size}

if use_cuda:
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform = transforms.Compose(
  [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

dataset1 = datasets.CIFAR10("./", train=True, download=True, transform=transform)
dataset1 = DatasetWrapper(dataset1)

dataset2 = datasets.CIFAR10("./", train=False, transform=transform)
dataset2 = DatasetWrapper(dataset2)

lwr = LWR(
        k=5,
        update_rate=0.9,
        num_batches_per_epoch=len(dataset1) // train_kwargs["batch_size"],
        dataset_length=len(dataset1),
        output_shape=(10,),
        tau=5,
        max_epochs=20,
        softmax_dim=1,
        use_kl=True,
    )

train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

model = Net().to(device)
snapshot = None

optimizer = optim.Adadelta(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
for epoch in range(1, epochs + 1):
    if (not usekl) and (epoch % k == 0):
        snapshot = deepcopy(model)

    train(
        usekl,        
        model,
        device,
        train_loader,
        optimizer,
        epoch,
        lwr,
        k,
        snapshot=snapshot,
    )
    test(model, device, test_loader)
    scheduler.step()

