# coding=utf-8


import torch.nn.functional as F


def cross_entropy(prediction, reference):
  return F.cross_entropy(prediction, reference, size_average=True)
