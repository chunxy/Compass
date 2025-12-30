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
def gen_zipf_distribution_float32(n, alpha=2):
  return np.random.zipf(alpha, n).astype(np.float32)


def generate_correlated_2d(n, correlation=0.5, mean=[0, 0], std=[1, 1], min_val=None, max_val=None):
  """
    Generate 2D correlated dataset using multivariate normal distribution.

    Parameters:
    - n: number of samples
    - correlation: correlation coefficient (between -1 and 1)
    - mean: mean values for each dimension [mean_x, mean_y]
    - std: standard deviations for each dimension [std_x, std_y]
    - min_val: minimum value for clipping (optional)
    - max_val: maximum value for clipping (optional)
    """
  # Create covariance matrix
  cov = np.array([[std[0]**2, correlation * std[0] * std[1]], [correlation * std[0] * std[1], std[1]**2]])

  # Generate correlated data
  data = np.random.multivariate_normal(mean, cov, size=n)

  # Clip to range if specified
  if min_val is not None or max_val is not None:
    data = np.clip(data, min_val, max_val)

  return data.astype(np.float32)


if __name__ == "__main__":
  np.random.seed(0)
  zipf_alpha, zipf_rg_ub = 2, 30
  for dataset, n in datasets.items():
    n_queries = dataset_nquery[dataset]
    data = gen_zipf_distribution_float32(n, zipf_alpha)
    data.tofile(f"/home/chunxy/repos/Compass/data/attr/{dataset}_1_{zipf_rg_ub}.skewed.value.bin")
    rg = np.zeros((n_queries, 2), dtype=np.float32)
    rg[:, 0] = np.random.uniform(0, zipf_rg_ub, n_queries)
    rg[:, 1] = np.random.uniform(rg[:, 0], zipf_rg_ub, n_queries)
    if (rg[:, 0] > rg[:, 1]).any():
      print("Error")
      exit()
    rg.tofile(f"/home/chunxy/repos/Compass/data/range/{dataset}_1_{zipf_rg_ub}.skewed.rg.bin")

    passrate = 0
    for i in range(n_queries):
      passrate += np.sum((data <= rg[i, 1]) & (data >= rg[i, 0])) / n
    print(f"{dataset} passrate: {passrate / n_queries}")

  variance, corr = 10, 0.5
  corr_rg_ub = 20
  for dataset, n in datasets.items():
    n_queries = dataset_nquery[dataset]
    # generate two-d datasets with correlation coefficient 0.5
    data = generate_correlated_2d(
      n,
      correlation=corr,
      mean=[0, 0],
      std=[variance, variance],
    )
    data.tofile(f"/home/chunxy/repos/Compass/data/attr/{dataset}_2_{corr_rg_ub}.correlated.value.bin")

    rg = np.zeros((n_queries * 2, 2), dtype=np.float32)
    rg[:n_queries, 0] = np.random.uniform(-corr_rg_ub, corr_rg_ub, n_queries)
    rg[:n_queries, 1] = np.random.uniform(-corr_rg_ub, corr_rg_ub, n_queries)
    rg[n_queries:, 0] = np.random.uniform(rg[:n_queries, 0], corr_rg_ub, n_queries)
    rg[n_queries:, 1] = np.random.uniform(rg[:n_queries, 1], corr_rg_ub, n_queries)
    if (rg[:n_queries, 0] > rg[:n_queries, 0]).any() or (rg[n_queries:, 1] > rg[n_queries:, 1]).any():
      print("Error")
      exit()
    rg.tofile(f"/home/chunxy/repos/Compass/data/range/{dataset}_2_{corr_rg_ub}.correlated.rg.bin")

    # Check actual correlation
    correlation_matrix = np.corrcoef(data[:, 0], data[:, 1])
    passrate = 0
    for i in range(n_queries):
      passrate += np.sum((data[:, 0] <= rg[n_queries:, 0]) & (data[:, 0] >= rg[:n_queries, 0]) & (data[:, 1] <= rg[n_queries:, 1])
                          & (data[:, 1] >= rg[:n_queries, 1])) / n
    print(f"{dataset} passrate: {passrate / n_queries}, correlation: {correlation_matrix[0, 1]:.4f}")
