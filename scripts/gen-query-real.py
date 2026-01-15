import numpy as np
from matplotlib import pyplot as plt
datasets = {
  "video-dedup": 1000000,
}

dataset_nquery = {
  "video-dedup": 10000,
}

MIN_PASS_RATIO = 0.001

if __name__ == "__main__":
  np.random.seed(0)

  for dataset, n in datasets.items():
    n_queries = dataset_nquery[dataset]

    real_ubs = [9003773, 1790325248, 1010082, 0]
    data = np.fromfile("/home/chunxy/repos/Compass/data/attr/real.bin", dtype=np.float32).reshape((-1, len(real_ubs)))
    indices = np.argsort(data[:, 0])
    data = data[indices][:, :2]
    # data.tofile(f"/home/chunxy/repos/Compass/data/attr/{dataset}_2_10000.real.value.bin")

    for i in range(len(real_ubs)):
      print(f"({data[:, i].min()}, {data[:, i].max()}) to {n} records.")
      if (data[:, i].size != n):
        print(f"Error: {dataset} data size mismatch: {data[:, i].size} != {n}")
        exit()

    rg = np.zeros((n_queries * 2, 2), dtype=np.float32)
    rg[:n_queries, 0] = np.random.uniform(0, np.quantile(data[:, 0], 0.9), n_queries)
    rg[:n_queries, 1] = np.random.uniform(0, np.quantile(data[:, 1], 0.9), n_queries)
    rg[n_queries:, 0] = np.random.uniform(rg[:n_queries, 0], real_ubs[0], n_queries)
    rg[n_queries:, 1] = np.random.uniform(rg[:n_queries, 1], real_ubs[1], n_queries)
    i = 0
    passrates = np.zeros(n_queries)
    while i < n_queries:
      pass_num = np.sum((data[:, 0] >= rg[i, 0]) & (data[:, 0] <= rg[i + n_queries, 0]) & (data[:, 1] >= rg[i, 1])
                        & (data[:, 1] <= rg[i + n_queries, 1]))
      if pass_num < MIN_PASS_RATIO * n:
        rg[i, 0] = (rg[i, 0]) / 2
        rg[i, 1] = (rg[i, 1]) / 2
        rg[i + n_queries, 0] = (rg[i + n_queries, 0] + real_ubs[0]) / 2
        rg[i + n_queries, 1] = (rg[i + n_queries, 1] + real_ubs[1]) / 2
        continue
      else:
        passrates[i] = pass_num / n
        i += 1
    # fix the upper bound
    # rg.astype(np.float32).tofile(f"/home/chunxy/repos/Compass/data/range/{dataset}_2_10000.real.rg.bin")
    plt.hist(passrates, bins=10)
    plt.savefig(f"/home/chunxy/repos/Compass/data/distrib/{dataset}_2_10000.real.passrate.png")
    print(f"Real: {dataset} passrate: {passrates.mean():.4f}")

