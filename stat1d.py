from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from config import *

STATS_DIR = "/home/chunxy/repos/Compass/stats"

prefix_name_map = {"": "KMeans", "bikmeans_": "Bisecting KMeans", "pca_": "PCA+KMeans"}
dataset_dpca = {"sift": 64, "audio": 64, "crawl": 128, "glove100": 64, "gist": 512, "video": 512}

def stat_top_k_in_cluster_by_selectivity():
  stat_dir = Path(STATS_DIR)
  stat_tmpl = "top_{}_in_cluster_hist_{}_{}_{}_{}.bin"
  pca_stat_tmpl = "top_{}_in_cluster_hist_{}_{}_{}_{}_{}.bin"
  nlist_s = [1000, 10000, 20000]
  ylim_s = [0.05, 0.01, 0.01]
  K = 10
  ranges = [(100, r) for r in (200, 600, 1100)]

  for i, dataset in enumerate(DATASETS):
    for nlist, ylim in zip(nlist_s, ylim_s):
      fig, axs = plt.subplots(len(ranges), len(prefix_name_map), layout='constrained')
      for j, rg in enumerate(ranges):
        for k, method in enumerate(prefix_name_map):
          if method == "pca_":
            stat_path = stat_dir / (method + pca_stat_tmpl.format(K, dataset, nlist, dataset_dpca[dataset], *rg))
          else:
            stat_path = stat_dir / (method + stat_tmpl.format(K, dataset, nlist, *rg))
          if not stat_path.exists(): continue
          with open(stat_path, "rb") as hist_file:
            data = list(int.from_bytes(hist_file.read(4), byteorder='little') for _ in range(nlist))
            bins = list(range(0, len(data) + 1))
            hist, _, _ = axs[j][k].hist(bins[:-1], bins, weights=data, density=True)
            # axs[i].vlines(50, ymin=0, ymax=0.05, linestyles='dashed', c='r')
            target = 0.9
            j_, cumu = 0, 0
            while j_ < len(hist):
              cumu += hist[j_]
              if cumu >= target:
                break
              j_ += 1
            axs[j][k].annotate(f"{target:.1%}, {j_}", xy=(j_, 0), xytext=(j_, ylim * 0.05), arrowprops=dict(facecolor='black', shrink=0.05))
            axs[j][k].set_ylim(0, ylim)
            axs[j][k].set_title(f"{dataset.upper()}-{(rg[1] - rg[0]) / 10000:.1%}-{prefix_name_map[method]}")
      fig.set_size_inches(20, 15)
      fig.savefig(f"figures_10/clustering/Clustering-{dataset.upper()}-{nlist}-Quality.jpg", dpi=100)
      plt.close()


def stat_cluster_imbalance_factor():
  stat_dir = Path(STATS_DIR)
  stat_tmpl = "cluster_element_count_{}_{}.bin"
  pca_stat_tmpl = "cluster_element_count_{}_{}_{}.bin"
  nlist_s = [1000, 10000, 20000]
  ylim_s = [0.01, 0.001, 0.001]

  for nlist, ylim in zip(nlist_s, ylim_s):
    fig, axs = plt.subplots(len(DATASETS), len(prefix_name_map), layout='constrained')
    for i, dataset in enumerate(DATASETS):
      for j, method in enumerate(prefix_name_map):
        if method == "pca_":
          stat_path = stat_dir / (method + pca_stat_tmpl.format(dataset, nlist, dataset_dpca[dataset]))
        else:
          stat_path = stat_dir / (method + stat_tmpl.format(dataset, nlist))
        if not stat_path.exists(): continue
        with open(stat_path, "rb") as hist_file:
          data = list(int.from_bytes(hist_file.read(4), byteorder='little') for _ in range(nlist))
          data.sort(reverse=True)
          bins = list(range(0, len(data) + 1))
          axs[i][j].hist(bins[:-1], bins, weights=data, density=True)
          axs[i][j].set_ylim(0, ylim)
          axs[i][j].set_title(f"{dataset.upper()}-{nlist}-{prefix_name_map[method]}")
    fig.set_size_inches(15, 20)
    fig.savefig(f"figures_10/clustering/Cluster-{nlist}-Imbalance.jpg", dpi=200)

stat_cluster_imbalance_factor()
stat_top_k_in_cluster_by_selectivity()
