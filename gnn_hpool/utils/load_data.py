# coding=utf-8

import networkx as nx
import numpy as np
import scipy as sc
import os
import re
import random
import logging

import torch
from torch.utils.data import Dataset, DataLoader

from gnn_hpool.utils import hparams_lib
from gnn_hpool.utils.global_variables import *


# follow a discussion here: https://github.com/RexYing/diffpool/issues/17
# no train-test split here


class GraphDataset(Dataset):

  def __init__(self, hparams, graph_list):
    self._hparams = hparams_lib.copy_hparams(hparams)
    self._device = torch.device(self._hparams.device)
    self.graph_list = []
    self.processed_graph_list = self.preprocess_graph(graph_list)

  def preprocess_graph(self, graph_list):
    processed_graph_list = []

    for graph in graph_list:
      graph_tmp_dict = {}

      # adjacency matrix
      adj = np.array(nx.to_numpy_matrix(graph))

      # node features
      node_tmp_feature = np.zeros((self._hparams.max_num_nodes, self._hparams.channel_list[0]))
      for index, feature in enumerate(graph.nodes()):
        # use the node label as input features
        # change the key here if the real features of node is wanted
        node_tmp_feature[index, :] = graph.node[index]['label']

      num_nodes = adj.shape[0]
      graph_tmp_dict[g_key.x] = torch.tensor(node_tmp_feature, dtype=torch.float32).to(self._device)
      graph_tmp_dict[g_key.y] = torch.tensor(graph.graph['label'], dtype=torch.long).to(self._device)
      graph_tmp_dict[g_key.node_num] = torch.tensor(num_nodes, dtype=torch.int16).to(self._device)
      graph_tmp_dict[g_key.adj_mat] = torch.zeros(self._hparams.max_num_nodes, self._hparams.max_num_nodes).to(self._device)
      graph_tmp_dict[g_key.adj_mat][:num_nodes, :num_nodes] = torch.tensor(adj, dtype=torch.float32).to(self._device)

      processed_graph_list.append(graph_tmp_dict)

    return processed_graph_list

  def __len__(self):
    return len(self.processed_graph_list)

  def __getitem__(self, idx):
    return self.processed_graph_list[idx]


class GraphDataLoaderWrapper(object):

  def __init__(self, hparams):

    self._hparams = hparams_lib.copy_hparams(hparams)
    self.graph_nx = read_graphfile(self._hparams.datadir, self._hparams.dataname, self._hparams.max_num_nodes)
    self.graph_count = len(self.graph_nx)
    self.val_size = self.graph_count // self._hparams.fold_num

  def get_loader(self, val_idx):
    graph_tmp = self.graph_nx
    random.shuffle(graph_tmp)

    train_graphs = graph_tmp[:val_idx * self.val_size]
    if val_idx < 9:
        train_graphs = train_graphs + graph_tmp[(val_idx+1) * self.val_size:]
    val_graphs = graph_tmp[val_idx * self.val_size: (val_idx + 1) * self.val_size]

    logging.info('\n * the length of training sets is {}; \n * the length of validation sets is {}'
                 .format(len(train_graphs), len(val_graphs)))

    training_set = GraphDataset(self._hparams, train_graphs)
    validation_set = GraphDataset(self._hparams, val_graphs)

    training_loader = DataLoader(
      training_set,
      batch_size=self._hparams.batch_size,
      shuffle=True,
    )

    validation_loader = DataLoader(
      validation_set,
      batch_size=self._hparams.batch_size,
      shuffle=False,
    )

    return training_loader, validation_loader


def read_graphfile(datadir, dataname, max_nodes=None):
  ''' Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
      graph index starts with 1 in file
  Returns:
      List of networkx objects with graph and node labels
  '''
  prefix = os.path.join(datadir, dataname, dataname)
  filename_graph_indic = prefix + '_graph_indicator.txt'
  # index of graphs that a given node belongs to
  graph_indic = {}
  with open(filename_graph_indic) as f:
    i = 1
    for line in f:
      line = line.strip("\n")
      graph_indic[i] = int(line)
      i += 1

  filename_nodes = prefix + '_node_labels.txt'
  node_labels = []
  try:
    with open(filename_nodes) as f:
      for line in f:
        line = line.strip("\n")
        node_labels += [int(line) - 1]
    num_unique_node_labels = max(node_labels) + 1
  except IOError:
    print('No node labels')

  filename_node_attrs = prefix + '_node_attributes.txt'
  node_attrs = []
  try:
    with open(filename_node_attrs) as f:
      for line in f:
        line = line.strip("\s\n")
        attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
        node_attrs.append(np.array(attrs))
  except IOError:
    print('No node attributes')

  label_has_zero = False
  filename_graphs = prefix + '_graph_labels.txt'
  graph_labels = []

  # assume that all graph labels appear in the dataset
  # (set of labels don't have to be consecutive)
  label_vals = []
  with open(filename_graphs) as f:
    for line in f:
      line = line.strip("\n")
      val = int(line)
      # if val == 0:
      #    label_has_zero = True
      if val not in label_vals:
        label_vals.append(val)
      graph_labels.append(val)
  # graph_labels = np.array(graph_labels)
  label_map_to_int = {val: i for i, val in enumerate(label_vals)}
  graph_labels = np.array([label_map_to_int[l] for l in graph_labels])
  # if label_has_zero:
  #    graph_labels += 1

  filename_adj = prefix + '_A.txt'
  adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
  index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
  num_edges = 0
  with open(filename_adj) as f:
    for line in f:
      line = line.strip("\n").split(",")
      e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
      adj_list[graph_indic[e0]].append((e0, e1))
      index_graph[graph_indic[e0]] += [e0, e1]
      num_edges += 1
  for k in index_graph.keys():
    index_graph[k] = [u - 1 for u in set(index_graph[k])]

  graphs = []
  for i in range(1, 1 + len(adj_list)):
    # indexed from 1 here
    G = nx.from_edgelist(adj_list[i])
    if max_nodes is not None and G.number_of_nodes() > max_nodes:
      continue

    # add features and labels
    G.graph['label'] = graph_labels[i - 1]
    for u in G.nodes():
      if len(node_labels) > 0:
        node_label_one_hot = [0] * num_unique_node_labels
        node_label = node_labels[u - 1]
        node_label_one_hot[node_label] = 1
        G.node[u]['label'] = node_label_one_hot
      if len(node_attrs) > 0:
        G.node[u]['feat'] = node_attrs[u - 1]
    if len(node_attrs) > 0:
      G.graph['feat_dim'] = node_attrs[0].shape[0]

    # relabeling
    mapping = {}
    it = 0
    if float(nx.__version__) < 2.0:
      for n in G.nodes():
        mapping[n] = it
        it += 1
    else:
      for n in G.nodes:
        mapping[n] = it
        it += 1

    # indexed from 0
    graphs.append(nx.relabel_nodes(G, mapping))
  return graphs
