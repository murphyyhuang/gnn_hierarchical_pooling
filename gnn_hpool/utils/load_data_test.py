# coding=utf-8

from gnn_hpool.utils.load_data import read_graphfile


def main():
  datadir = '/home/murphyhuang/dev/src/github.com/EstelleHuang666/gnn_hierarchical_pooling/data/gnn_enzymes_source_20190905'
  dataname = 'ENZYMES'

  data_list = read_graphfile(datadir, dataname)

  print(data_list)


if __name__ == '__main__':
  main()
