# coding=utf-8

import torch
from torch.nn.modules.module import Module
import torch.nn.functional as F

from gnn_hpool.utils.global_variables import g_key
from gnn_hpool.utils import hparams_lib
from gnn_hpool.models.gcn_hpool_submodel import GcnHpoolSubmodel
from gnn_hpool.layers import gcn_layer


class GcnHpoolEncoder(Module):

  def __init__(self, hparams):
    super(GcnHpoolEncoder, self).__init__()

    self._hparams = hparams_lib.copy_hparams(hparams)
    self.build_graph()
    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, gcn_layer.GraphConvolution):
        m.weight.data = torch.nn.init.xavier_uniform(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
        if m.bias is not None:
          m.bias.data = torch.nn.init.constant(m.bias.data, 0.0)

  def build_graph(self):

    # entry GCN
    self.entry_conv_first = gcn_layer.GraphConvolution(
      in_features=self._hparams.channel_list[0],
      out_features=self._hparams.channel_list[1],
      hparams=self._hparams,
    )
    self.entry_conv_block = gcn_layer.GraphConvolution(
      in_features=self._hparams.channel_list[1],
      out_features=self._hparams.channel_list[1],
      hparams=self._hparams,
    )
    self.entry_conv_last = gcn_layer.GraphConvolution(
      in_features=self._hparams.channel_list[1],
      out_features=self._hparams.channel_list[2],
      hparams=self._hparams,
    )

    self.gcn_hpool_layer = GcnHpoolSubmodel(
      self._hparams.channel_list[2], self._hparams.channel_list[3], self._hparams.channel_list[4],
      self._hparams.node_list[0], self._hparams.node_list[1], self._hparams.node_list[2],
      self._hparams
    )

    self.pred_model = torch.nn.Sequential([
      torch.nn.Linear(2 * 3 * self.channel_list[-3], self._hparams.channel_list[-2]),
      torch.nn.ReLU,
      torch.nn.Linear(self._hparams.channel_list[-2], self._hparams.channel_list[-1])
    ])

  def forward(self, graph_input):

    node_feature = graph_input[g_key.x]
    adjacency_mat = graph_input[g_key.adj_mat]
    batch_num_nodes = graph_input[g_key.node_num]

    # input mask
    max_num_nodes = adjacency_mat.size()[1]
    embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)

    # entry embedding gcn
    embedding_tensor_1 = self.gcn_forward(
      node_feature, adjacency_mat,
      self.entry_conv_first, self.entry_conv_block, self.entry_conv_last,
      embedding_mask
    )
    output_1, _ = torch.max(embedding_tensor_1, dim=1)

    # hpool layer
    output_2, _, _, _ = self.gcn_hpool_layer(
      embedding_tensor_1, node_feature, adjacency_mat, batch_num_nodes
    )

    output = torch.cat([output_1, output_2], dim=1)
    ypred = self.pred_model(output)

    return ypred

  def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):
    out_all = []

    layer_out_1 = F.relu(conv_first(x, adj))
    layer_out_1 = self.apply_bn(layer_out_1)
    out_all.append(layer_out_1)

    layer_out_2 = F.relu(conv_block(layer_out_1, adj))
    layer_out_2 = self.apply_bn(layer_out_2)
    out_all.append(layer_out_2)

    layer_out_3 = F.relu(conv_last(layer_out_2, adj))
    out_all.append(layer_out_3)
    out_all = torch.cat(out_all, dim=2)
    if embedding_mask is not None:
      out_all = out_all * embedding_mask

    return out_all

  def apply_bn(self, x):
      ''' Batch normalization of 3D tensor x
      '''
      bn_module = torch.nn.BatchNorm1d(x.size()[1])
      return bn_module(x)

  @staticmethod
  def construct_mask(max_nodes, batch_num_nodes):
      ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
      corresponding column are 1's, and the rest are 0's (to be masked out).
      Dimension of mask: [batch_size x max_nodes x 1]
      '''
      # masks
      packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
      batch_size = len(batch_num_nodes)
      out_tensor = torch.zeros(batch_size, max_nodes)
      for i, mask in enumerate(packed_masks):
          out_tensor[i, :batch_num_nodes[i]] = mask
      return out_tensor.unsqueeze(2)
