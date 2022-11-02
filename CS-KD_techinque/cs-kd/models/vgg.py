import torch.nn as nn
from torchvision import models


__all__ = ['CIFAR10_VGG16', 'CIFAR100_VGG16']

def CIFAR10_VGG16():
    num_classes = 10
    model = models.vgg16()
    model.classifier[6] = nn.Linear(4096,num_classes)
    return model

def CIFAR100_VGG16():
    num_classes = 100
    model = models.vgg16()
    model.classifier[6] = nn.Linear(4096,num_classes)
    return model
