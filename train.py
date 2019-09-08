# coding=utf-8

import os
import torch
import numpy as np
import logging
import argparse

from gnn_hpool.utils import hparam


def main(args):

  hparams = hparam.HParams()
  hparams.from_yaml(args.hparam_path)

  # reproducibility
  if hparams.device == 'cuda':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  torch.manual_seed(1024)
  np.random.seed(1024)

  # set default GPU
  os.environ['CUDA_VISIBLE_DEVICES'] = hparams.cuda_visible_devices

  from gnn_hpool.bin import train_eval
  train_eval.train_eval(hparams)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  parser = argparse.ArgumentParser(description='Parameters for the training of GNN')
  parser.add_argument('--hparam_path', nargs='?', type=str,
                      default='./config/hparams_testdb.yml',
                      help='The path to .yml file which contains all the hyperparameters.'
                      )

  args = parser.parse_args()
  main(args)
