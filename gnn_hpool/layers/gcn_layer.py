# coding=utf-8

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
  """
  Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
  """

  def __init__(self, in_features, out_features, hparams, bias=True):
    super(GraphConvolution, self).__init__()
    self._hparams = hparams
    self.in_features = in_features
    self.out_features = out_features
    self.weight = Parameter(torch.FloatTensor(in_features, out_features))
    if bias:
      self.bias = Parameter(torch.FloatTensor(out_features))
    else:
      self.register_parameter('bias', None)
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
      self.bias.data.uniform_(-stdv, stdv)

  def norm(self, adj):
    node_num = adj.shape[-1]
    # add remaining self-loops
    self_loop = torch.eye(node_num).to(self._hparams.device)
    self_loop = self_loop.reshape((1, node_num, node_num))
    self_loop = self_loop.repeat(adj.shape[0], 1, 1)
    adj_post = adj + self_loop
    # signed adjacent matrix
    deg_abs = torch.sum(torch.abs(adj_post), dim=-1)
    deg_abs_sqrt = deg_abs.pow(-0.5)
    diag_deg = torch.diag_embed(deg_abs_sqrt, dim1=-2, dim2=-1)

    norm_adj = torch.matmul(torch.matmul(diag_deg, adj_post), diag_deg)
    return norm_adj

  def forward(self, input, adj):
    support = torch.matmul(input, self.weight)
    adj_norm = self.norm(adj)
    output = torch.matmul(support.transpose(1, 2), adj_norm.transpose(1, 2))
    output = output.transpose(1, 2)
    if self.bias is not None:
      return output + self.bias
    else:
      return output

  def __repr__(self):
    return self.__class__.__name__ + ' (' \
           + str(self.in_features) + ' -> ' \
           + str(self.out_features) + ')'
