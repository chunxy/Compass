from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

dataset_config = {
  "sift": [(500, 32), (500, 64), (1000, 32), (1000, 64), (10000, 64)],
  "audio": [(500, 32), (500, 64), (1000, 32), (1000, 64), (10000, 64)],
  "glove100": [(500, 32), (500, 64), (1000, 32), (1000, 64), (1000, 50)],
  "crawl": [],
  "gist": [(500, 32), (500, 64), (500, 128), (1000, 32), (1000, 64), (1000, 128)],
  "video": [(500, 32), (500, 64), (500, 128), (1000, 32), (1000, 64), (1000, 128)],
}
dataset_nb = {
  "sift": 1_000_000,
  "audio": 1_000_000,
  "glove100": 1_000_000,
  "gist": 1_183_514,
  "crawl": 2_000_000,
  "video": 1_000_000,
}


def stat_top_k_in_cluster_by_selectivity():
  STATS_DIR = "/home/chunxy/repos/Compass/stats"
  stat_dir = Path(STATS_DIR)
  stat_tmpl = "proclus_top_{}_in_cluster_hist_{}_{}_{}_{}_{}.bin"
  k = 10
  ranges = [(0, 10000), *[(100, r) for r in (200, 300, 600)], *[(100, r) for r in range(1100, 10000, 1000)]]

  for d in dataset_config.keys():
    for rg in ranges:
      fig, axs = plt.subplots(1, max(2, len(dataset_config[d])), layout='constrained')
      for i, config in enumerate(dataset_config[d]):
        nlist, dim = config
        stat_path = stat_dir / stat_tmpl.format(k, d, nlist, dim, *rg)
        if not stat_path.exists(): continue
        with open(stat_path, "rb") as hist_file:
          data = list(int.from_bytes(hist_file.read(4), byteorder='little') for _ in range(nlist))
          bins = list(range(0, len(data) + 1))
          hist, _, _ = axs[i].hist(bins[:-1], bins, weights=data, density=True)
          # axs[i].vlines(50, ymin=0, ymax=0.05, linestyles='dashed', c='r')
          target = 0.9
          j, cumu = 0, 0
          while j < len(hist):
            cumu += hist[j]
            if cumu >= target:
              break
            j += 1
          axs[i].annotate(f"{target:.1%}, {j}", xy=(j, 0), xytext=(j, 0.05), arrowprops=dict(facecolor='black', shrink=0.05))
          axs[i].set_ylim(0, 0.5)
          axs[i].set_title(f"{d.upper()}-{rg}-{nlist}-{dim}")
      fig.set_size_inches(20, 5)
      fig.savefig(f"figures_10/clustering/{d.upper()}-Proclus-{(rg[1] - rg[0]) / 10000:.1%}-Clustering-Quality.jpg", dpi=100)
      plt.close()


def stat_cluster_imbalance_factor():
  STATS_DIR = "/home/chunxy/repos/Compass/checkpoints/Proclus"
  stat_dir = Path(STATS_DIR)
  for d, configs in dataset_config.items():
    fig, axs = plt.subplots(1, max(2, len(configs)), layout='constrained')
    axs = axs.flat
    for i, c in enumerate(configs):
      nlist, dproclus = c
      stat_tmpl = f"{nlist}-{dproclus}.ranking"
      stat_path = stat_dir / d / stat_tmpl
      if not stat_path.exists(): continue
      with open(stat_path, "rb") as assignment_file:
        assigned_clusters = list(int.from_bytes(assignment_file.read(8), byteorder='little') for _ in range(dataset_nb[d]))
        counts = np.zeros(nlist)
        for cluster in assigned_clusters:
          counts[cluster] += 1
        counts.sort()
        counts = counts[::-1]
        bins = list(range(0, nlist + 1))
        axs[i].hist(bins[:-1], bins, weights=counts, density=True)
        axs[i].set_ylim(0, 0.015)
        axs[i].set_title(f"{d.upper()}-{nlist}-{dproclus}")
    fig.set_size_inches(20, 5)
    fig.savefig(f"figures_10/clustering/{d.upper()}-Proclus-Cluster-Imbalance.jpg", dpi=200)
    plt.close(fig)


stat_cluster_imbalance_factor()
stat_top_k_in_cluster_by_selectivity()
