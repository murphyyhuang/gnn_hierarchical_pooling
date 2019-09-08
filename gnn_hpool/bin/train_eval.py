# coding=utf-8

import os
import time
import logging
import matplotlib

try:
  import matplotlib.pyplot as plt
except ModuleNotFoundError:
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

import numpy as np
import networkx as nx

import torch
import tensorboardX
from gnn_hpool.utils import get_loss
from gnn_hpool.utils import common_utils
from gnn_hpool.utils.global_variables import *
from gnn_hpool.utils.evaluate import evaluate
from gnn_hpool.utils import load_data
from gnn_hpool.models import gcn_hpool_encoder


def train_eval(hparams):
  data_loader = load_data.GraphDataLoaderWrapper(hparams)

  all_vals = []
  for val_idx in range(hparams.fold_num):
    logging.warning('* validation index: {}'.format(val_idx))
    training_loader, validation_loader = data_loader.get_loader(val_idx)
    summary_writer = tensorboardX.SummaryWriter(
      logdir=os.path.join(hparams.model_save_path, str(hparams.timestamp) + '/val_{}'.format(val_idx))
    )

    model = gcn_hpool_encoder.GcnHpoolEncoder(hparams).to(torch.device(hparams.device))
    _, val_accs = train_eval_iter(model, training_loader, validation_loader, summary_writer, hparams)
    all_vals.append(np.array(val_accs))

  all_vals = np.vstack(all_vals)
  all_vals = np.mean(all_vals, axis=0)
  logging.warning('* all of the validation results: ', all_vals)
  logging.warning('* the best validation results & its id: {} @ {}'.format(p.max(all_vals), np.argmax(all_vals)))


def train_eval_iter(model, train_dataset, eval_dataset, writer, hparams):
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hparams.learning_rate)

  best_val_result = {
    'epoch': 0,
    'loss': 0,
    'acc': 0}
  train_accs = []
  train_epochs = []
  best_val_accs = []
  best_val_epochs = []
  val_accs = []

  for epoch in range(hparams.epoch):
    total_time = 0
    avg_loss = 0.0
    model.train()

    for batch_idx, graph_data in enumerate(train_dataset):

      begin_time = time.time()
      optimizer.zero_grad()

      # run model
      ypred = model(graph_data)
      loss = get_loss.cross_entropy(ypred, graph_data[g_key.y])
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip)
      optimizer.step()

      # record
      avg_loss += loss
      elapsed = time.time() - begin_time
      total_time += elapsed

      # log once per XX epochs
      if epoch % 10 == 0 and batch_idx == len(train_dataset) // 2 and writer is not None:
        log_assignment(model.gcn_hpool_layer.pool_tensor, writer, epoch, writer_batch_idx)
        log_graph(graph_data[g_key.adj_mat], graph_data[g_key.node_num], writer, epoch, writer_batch_idx, model.gcn_hpool_layer.pool_tensor)

    avg_loss /= batch_idx + 1
    if writer is not None:
      writer.add_scalar('loss/avg_loss', avg_loss, epoch)

    result = evaluate(train_dataset, model, hparams, max_num_examples=100)
    train_accs.append(result['acc'])
    train_epochs.append(epoch)

    val_result = evaluate(eval_dataset, model, hparams)
    val_accs.append(val_result['acc'])
    if val_result['acc'] > best_val_result['acc'] - 1e-7:
      best_val_result['acc'] = val_result['acc']
      best_val_result['epoch'] = epoch
      best_val_result['loss'] = avg_loss
    if writer is not None:
      writer.add_scalar('acc/train_acc', result['acc'], epoch)
      writer.add_scalar('acc/val_acc', val_result['acc'], epoch)
      writer.add_scalar('loss/best_val_loss', best_val_result['loss'], epoch)

    logging.warning('Best val result: {:.4f} @ epoch {}'.format(best_val_result['acc'], best_val_result['epoch']))
    best_val_epochs.append(best_val_result['epoch'])
    best_val_accs.append(best_val_result['acc'])

  matplotlib.style.use('seaborn')
  plt.switch_backend('agg')
  plt.figure()
  plt.plot(train_epochs, common_utils.exp_moving_avg(train_accs, 0.85), '-', lw=1)

  plt.plot(best_val_epochs, best_val_accs, 'bo')
  plt.legend(['train', 'val'])
  plt.savefig(os.path.join(hparams.model_save_path, str(hparams.timestamp) + '.png'), dpi=600)
  plt.close()
  matplotlib.style.use('default')

  return model, val_accs


def log_assignment(assign_tensor, writer, epoch, batch_idx):
  plt.switch_backend('agg')
  fig = plt.figure(figsize=(8, 6), dpi=300)

  # has to be smaller than args.batch_size
  for i in range(len(batch_idx)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(assign_tensor.cpu().data.numpy()[batch_idx[i]], cmap=plt.get_cmap('BuPu'))
    cbar = plt.colorbar()
    cbar.solids.set_edgecolor("face")
  plt.tight_layout()
  fig.canvas.draw()

  # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  data = tensorboardX.utils.figure_to_image(fig)
  writer.add_image('assignment', data, epoch)


def log_graph(adj, batch_num_nodes, writer, epoch, batch_idx, assign_tensor=None):
  plt.switch_backend('agg')
  fig = plt.figure(figsize=(8, 6), dpi=300)

  for i in range(len(batch_idx)):
    ax = plt.subplot(2, 2, i + 1)
    num_nodes = batch_num_nodes[batch_idx[i]]
    adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()
    G = nx.from_numpy_matrix(adj_matrix)
    nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color='#336699',
            edge_color='grey', width=0.5, node_size=300,
            alpha=0.7)
    ax.xaxis.set_visible(False)

  plt.tight_layout()
  fig.canvas.draw()

  data = tensorboardX.utils.figure_to_image(fig)
  writer.add_image('graphs', data, epoch)

  assignment = assign_tensor.cpu().data.numpy()
  fig = plt.figure(figsize=(8, 6), dpi=300)

  num_clusters = assignment.shape[2]
  all_colors = np.array(range(num_clusters))

  for i in range(len(batch_idx)):
    ax = plt.subplot(2, 2, i + 1)
    num_nodes = batch_num_nodes[batch_idx[i]]
    adj_matrix = adj[batch_idx[i], :num_nodes, :num_nodes].cpu().data.numpy()

    label = np.argmax(assignment[batch_idx[i]], axis=1).astype(int)
    label = label[: batch_num_nodes[batch_idx[i]]]
    node_colors = all_colors[label]

    G = nx.from_numpy_matrix(adj_matrix)
    nx.draw(G, pos=nx.spring_layout(G), with_labels=False, node_color=node_colors,
            edge_color='grey', width=0.4, node_size=50, cmap=plt.get_cmap('Set1'),
            vmin=0, vmax=num_clusters - 1,
            alpha=0.8)

  plt.tight_layout()
  fig.canvas.draw()

  # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  data = tensorboardX.utils.figure_to_image(fig)
  writer.add_image('graphs_colored', data, epoch)
