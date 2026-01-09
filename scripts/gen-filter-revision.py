import numpy as np

datasets = {
  "sift-dedup": 1000000 - 14538,
  "audio-dedup": 1000000,
  "gist-dedup": 1000000 - 17306,
  "video-dedup": 1000000,
  "glove100": 1183514,
  "crawl": 1989995,
}

dataset_nquery = {
  "sift-dedup": 10000,
  "audio-dedup": 10000,
  "gist-dedup": 1000,
  "video-dedup": 10000,
  "glove100": 10000,
  "crawl": 10000,
}

# generate zipf distribution data for float32
def gen_zipf_distribution_int32(n, alpha=2):
  return np.random.zipf(alpha, n).astype(np.int32)

if __name__ == "__main__":
  np.random.seed(0)

  for dataset, n in datasets.items():
    n_queries = dataset_nquery[dataset]

    zipf_rg_ub = 30
    # data = gen_zipf_distribution_int32(n, 2)
    # data.astype(np.float32).tofile(f"/home/chunxy/repos/Compass/data/attr/{dataset}_1_{zipf_rg_ub}.newfilter.value.bin")

    data = np.fromfile(f"/home/chunxy/repos/Compass/data/attr/{dataset}_1_{zipf_rg_ub}.skewed.value.bin", dtype=np.float32)
    data = data.astype(np.int32)
    print(f"({data.min()}, {data.max()}) to {n} records.")
    if (data.size != n):
      print(f"Error: {dataset} data size mismatch: {data.size} != {n}")
      exit()

    rg = np.random.randint(1, zipf_rg_ub, n_queries, dtype=np.int32)
    i = 0
    while i < n_queries:
      pass_num = np.sum(data >= rg[i])
      if pass_num < 100:
        rg[i] //= 2
        continue
      else:
        i += 1
    rg.astype(np.float32).tofile(f"/home/chunxy/repos/Compass/data/range/{dataset}_1_{zipf_rg_ub}.onesided.rg.bin")
    passrate = 0
    for i in range(n_queries):
      passrate += np.sum(data >= rg[i]) / n
    print(f"Onesided: {dataset} passrate: {passrate / n_queries}")


    point_ub = 10
    rg = np.random.randint(1, point_ub, n_queries, dtype=np.int32)
    i = 0
    while i < n_queries:
      pass_num = np.sum((data == rg[i]))
      if pass_num < 100:
        rg[i] //= 2
        continue
      else:
        i += 1
    rg.astype(np.float32).tofile(f"/home/chunxy/repos/Compass/data/range/{dataset}_1_{zipf_rg_ub}.point.rg.bin")

    passrate = 0
    for i in range(n_queries):
      passrate += np.sum((data == rg[i])) / n
    print(f"Point: {dataset} passrate: {passrate / n_queries}")


    rg = np.random.randint(1, point_ub, n_queries, dtype=np.int32)
    i = 0
    while i < n_queries:
      pass_num = np.sum((data != rg[i]))
      if pass_num < 100:
        rg[i] *= 2
        continue
      else:
        i += 1
    rg.astype(np.float32).tofile(f"/home/chunxy/repos/Compass/data/range/{dataset}_1_{zipf_rg_ub}.negation.rg.bin")

    passrate = 0
    for i in range(n_queries):
      passrate += np.sum((data != rg[i])) / n
    print(f"Negation: {dataset} passrate: {passrate / n_queries}")
