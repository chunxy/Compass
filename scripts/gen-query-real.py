import numpy as np

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

    real_ub = 30
    data = np.fromfile("/home/chunxy/repos/Compass/data/attr/real.value.bin", dtype=np.float32)
    indices = np.argsort(data)
    data = data[indices]
    # fix the upper bound
    data.tofile(f"/home/chunxy/repos/Compass/data/attr/{dataset}_2_10000.real.value.bin", dtype=np.float32)

    print(f"({data.min()}, {data.max()}) to {n} records.")
    if (data.size != n):
      print(f"Error: {dataset} data size mismatch: {data.size} != {n}")
      exit()

    rg = np.zeros((n_queries * 2, 2), dtype=np.float32)
    rg[:n_queries, 0] = np.random.uniform(0, real_ub, n_queries)
    rg[:n_queries, 1] = np.random.uniform(0, real_ub, n_queries)
    rg[n_queries:, 0] = np.random.uniform(rg[:n_queries, 0], real_ub, n_queries)
    rg[n_queries:, 1] = np.random.uniform(rg[:n_queries, 1], real_ub, n_queries)
    i = 0
    while i < n_queries:
      pass_num = np.sum((data[:, 0] >= rg[i, 0]) & (data[:, 0] <= rg[i + n_queries, 0]) & (data[:, 1] >= rg[i, 1])
                        & (data[:, 1] <= rg[i + n_queries, 1]))
      if pass_num < MIN_PASS_RATIO * n:
        rg[i, 0] = (rg[i, 0]) / 2
        rg[i, 1] = (rg[i, 1]) / 2
        rg[i + n_queries, 0] = (rg[i + n_queries, 0] + real_ub) / 2
        rg[i + n_queries, 1] = (rg[i + n_queries, 1] + real_ub) / 2
        continue
      else:
        i += 1
    # fix the upper bound
    rg.astype(np.float32).tofile(f"/home/chunxy/repos/Compass/data/range/{dataset}_2_10000.real.rg.bin")
    passrate = 0
    for i in range(n_queries):
      passrate += np.sum((data[:, 0] >= rg[i, 0]) & (data[:, 0] <= rg[i + n_queries, 0]) & (data[:, 1] >= rg[i, 1])
                          & (data[:, 1] <= rg[i + n_queries, 1])) / n
    print(f"Real: {dataset} passrate: {passrate / n_queries}")

