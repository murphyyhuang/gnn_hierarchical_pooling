# coding=utf-8

import torch
from torch.nn.modules.module import Module
import torch.nn.functional as F

from gnn_hpool.layers.hierarchical_diff_pooling import dense_diff_pool
from gnn_hpool.utils import hparams_lib
from gnn_hpool.layers import gcn_layer


class GcnHpoolSubmodel(Module):
  def __init__(self, in_feature, hidden_feature, out_feature, in_node, hidden_node, out_node, hparams):
    super(GcnHpoolSubmodel, self).__init__()

    self._hparams = hparams_lib.copy_hparams(hparams)
    self.build_graph(in_feature, hidden_feature, out_feature, in_node, hidden_node, out_node)
    self.reset_parameters()

    self.pool_tensor = None

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, gcn_layer.GraphConvolution):
        m.weight.data = torch.nn.init.xavier_uniform(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
        if m.bias is not None:
          m.bias.data = torch.nn.init.constant(m.bias.data, 0.0)

  def build_graph(self, in_feature, hidden_feature, out_feature, in_node, hidden_node, out_node):

    # embedding blocks

    self.embed_conv_first = gcn_layer.GraphConvolution(
      in_features=in_feature,
      out_features=hidden_feature,
      hparams=self._hparams,
    )
    self.embed_conv_block = gcn_layer.GraphConvolution(
      in_features=hidden_feature,
      out_features=hidden_feature,
      hparams=self._hparams,
    )
    self.embed_conv_last = gcn_layer.GraphConvolution(
      in_features=hidden_feature,
      out_features=out_feature,
      hparams=self._hparams,
    )

    # pooling blocks

    self.pool_conv_first = gcn_layer.GraphConvolution(
      in_features=in_node,
      out_features=hidden_node,
      hparams=self._hparams,
    )
    self.pool_conv_block = gcn_layer.GraphConvolution(
      in_features=hidden_node,
      out_features=hidden_node,
      hparams=self._hparams,
    )
    self.pool_conv_last = gcn_layer.GraphConvolution(
      in_features=hidden_node,
      out_features=out_node,
      hparams=self._hparams,
    )

    self.pool_linear = torch.nn.Linear(hidden_node * 2 + out_node, out_node)

  def forward(self, embedding_tensor, pool_x_tensor, adj, embedding_mask):

    pooling_tensor = self.gcn_forward(
      pool_x_tensor, adj,
      self.pool_conv_first, self.pool_conv_block, self.pool_conv_last,
      embedding_mask
    )
    pooling_tensor = F.softmax(self.pool_linear(pooling_tensor), dim=-1)
    if embedding_mask is not None:
      pooling_tensor = pooling_tensor * embedding_mask

    x_pool, adj_pool = dense_diff_pool(embedding_tensor, adj, pooling_tensor)

    embedding_tensor = self.gcn_forward(
      x_pool, adj_pool,
      self.embed_conv_first, self.embed_conv_block, self.embed_conv_last,
    )

    output, _ = torch.max(embedding_tensor, dim=1)

    self.pool_tensor = pooling_tensor
    return output, adj_pool, x_pool, embedding_tensor

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
