# coding=utf-8

""" HParams handling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gnn_hpool.utils import hparam


def copy_hparams(hparams):
  hp_vals = hparams.values()
  new_hparams = hparam.HParams(**hp_vals)
  return new_hparams


def create_hparams(config_dir):
  hparams = hparam.HParams()
  hparams.from_yaml(config_dir)

  return hparams
