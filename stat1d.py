from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from config import *

STATS_DIR = "/home/chunxy/repos/Compass/stats"


def stat_top_k_in_cluster_by_selectivity():
  stat_dir = Path(STATS_DIR)
  stat_tmpl = "top_{}_in_cluster_hist_{}_{}_{}_{}.bin"
  ranges = [(0, 10000), *[(100, r) for r in (200, 300, 600)], *[(100, r) for r in range(1100, 10000, 1000)]]
  nlist = 1000
  k = 10

  for rg in ranges:
    fig, axs = plt.subplots(1, len(DATASETS), layout='constrained')
    for i, dataset in enumerate(DATASETS):
      stat_path = stat_dir / stat_tmpl.format(k, dataset, nlist, *rg)
      with open(stat_path, "rb") as hist_file:
        data = list(int.from_bytes(hist_file.read(4), byteorder='little') for _ in range(nlist))
        data = data[:200]
        bins = list(range(0, len(data) + 1))
        axs[i].hist(bins[:-1], bins, weights=data, density=True)
        # axs[i].vlines(50, ymin=0, ymax=0.5, linestyles='dashed', c='r')
        axs[i].annotate('50', xy=(50, 0), xytext=(50, 0.05), arrowprops=dict(facecolor='black', shrink=0.05))
        axs[i].set_ylim(0, 0.5)
        axs[i].set_title(f"{dataset.upper()}-{rg}")
    fig.set_size_inches(20, 5)
    fig.savefig(f"figures_10/All-{(rg[1] - rg[0]) / 10000:.1%}-Clustering-Quality.jpg", dpi=100)


def stat_cluster_imbalance_factor():
  stat_dir = Path(STATS_DIR)
  stat_tmpl = "cluster_element_count_{}_{}.bin"
  nlist = 1000

  fig, axs = plt.subplots(1, len(DATASETS), layout='constrained')
  for i, dataset in enumerate(DATASETS):
    stat_path = stat_dir / stat_tmpl.format(dataset, nlist)
    with open(stat_path, "rb") as hist_file:
      data = list(int.from_bytes(hist_file.read(4), byteorder='little') for _ in range(1000))
      data.sort(reverse=True)
      bins = list(range(0, len(data) + 1))
      axs[i].hist(bins[:-1], bins, weights=data, density=True)
      axs[i].set_ylim(0, 0.03)
      axs[i].set_title(f"{dataset.upper()}-{nlist}")
  fig.set_size_inches(20, 5)
  fig.savefig(f"figures_10/All-Cluster-Imbalance.jpg", dpi=200)


stat_top_k_in_cluster_by_selectivity()
stat_cluster_imbalance_factor()
