# coding=utf-8

import torch
import numpy as np
import sklearn.metrics as metrics

from gnn_hpool.utils.global_variables import *


def evaluate(dataset, model, hparams, max_num_examples=None):
  model.eval()

  labels = []
  preds = []
  for batch_idx, graph in enumerate(dataset):

    ypred = model(graph)
    _, indices = torch.max(ypred, 1)
    preds.append(indices.cpu().detach().numpy())
    labels.append(graph[g_key.y].cpu().detach().numpy())

    if max_num_examples is not None:
      if (batch_idx + 1) * hparams.batch_size > max_num_examples:
        break

  labels = np.hstack(labels)
  preds = np.hstack(preds)

  result = {'prec': metrics.precision_score(labels, preds, average='macro'),
            'recall': metrics.recall_score(labels, preds, average='macro'),
            'acc': metrics.accuracy_score(labels, preds),
            'F1': metrics.f1_score(labels, preds, average="micro")}
  return result
