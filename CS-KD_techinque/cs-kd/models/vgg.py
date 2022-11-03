import torch.nn as nn
from torchvision import models


__all__ = ['CIFAR10_VGG16', 'CIFAR100_VGG16']

def CIFAR10_VGG16(pretrained, num_classes, **kwargs):
    model = models.vgg16_bn(weights='DEFAULT', **kwargs)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs,num_classes)
    return model

def CIFAR100_VGG16(pretrained, num_classes, **kwargs):
    model = models.vgg16_bn(weights='DEFAULT', **kwargs)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs,num_classes)
    return model
