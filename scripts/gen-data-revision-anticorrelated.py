import numpy as np
from matplotlib import pyplot as plt

datasets = {
  "sift-dedup": 1000000 - 14538,
  "audio-dedup": 1000000,
  "gist-dedup": 1000000 - 17306,
  "video-dedup": 1000000,
  "glove100": 1183514,
  "crawl": 1989995,
  "flickr": 4203901,
  "deep10m": 10000000,
}

dataset_nquery = {
  "sift-dedup": 10000,
  "audio-dedup": 10000,
  "gist-dedup": 1000,
  "video-dedup": 10000,
  "glove100": 10000,
  "crawl": 10000,
  "flickr": 29999,
  "deep10m": 10000,
}


# generate zipf distribution data for float32
def gen_zipf_distribution_float32(n, alpha=2):
  return np.random.zipf(alpha, n).astype(np.float32)


MIN_PASS_RATIO = 0.001


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
  variance, corr = 10, -0.5
  corr_rg_ub = 20
  for dataset, n in datasets.items():
    n_queries = dataset_nquery[dataset]
    # generate two-d datasets with correlation coefficient -0.5
    data = generate_correlated_2d(
      n,
      correlation=corr,
      mean=[0, 0],
      std=[variance, variance],
    )
    indices = np.argsort(data[:, 0])
    data = data[indices]
    # data.tofile(f"/home/chunxy/repos/Compass/data/attr/{dataset}_2_{corr_rg_ub}.anticorrelated.value.bin")
    passrates = np.zeros(n_queries)

    rg = np.zeros((n_queries * 2, 2), dtype=np.float32)
    rg[:n_queries, 0] = np.random.uniform(-corr_rg_ub, corr_rg_ub, n_queries)
    rg[:n_queries, 1] = np.random.uniform(-corr_rg_ub, corr_rg_ub, n_queries)
    rg[n_queries:, 0] = np.random.uniform(rg[:n_queries, 0], corr_rg_ub, n_queries)
    rg[n_queries:, 1] = np.random.uniform(rg[:n_queries, 1], corr_rg_ub, n_queries)
    if (rg[:n_queries, 0] > rg[n_queries:, 0]).any() or (rg[:n_queries, 1] > rg[n_queries:, 1]).any():
      print("Error")
      exit()
    i = 0
    while i < n_queries:
      pass_num = np.sum((data[:, 0] >= rg[i, 0]) & (data[:, 0] <= rg[i + n_queries, 0]) & (data[:, 1] >= rg[i, 1])
                        & (data[:, 1] <= rg[i + n_queries, 1]))
      if pass_num < MIN_PASS_RATIO * n:
        rg[i, 0] = (rg[i, 0] - corr_rg_ub) / 2
        rg[i, 1] = (rg[i, 1] - corr_rg_ub) / 2
        rg[i + n_queries, 0] = (rg[i + n_queries, 0] + corr_rg_ub) / 2
        rg[i + n_queries, 1] = (rg[i + n_queries, 1] + corr_rg_ub) / 2
        continue
      else:
        passrates[i] = pass_num / n
        i += 1
    # rg.tofile(f"/home/chunxy/repos/Compass/data/range/{dataset}_2_{corr_rg_ub}.anticorrelated.rg.bin")

    # Check actual correlation
    correlation_matrix = np.corrcoef(data[:, 0], data[:, 1])
    print(f"{dataset} passrate: {passrates.mean():.4f}, correlation: {correlation_matrix[0, 1]:.4f}")
    plt.hist(passrates, bins=20, weights=np.ones_like(passrates) / passrates.size)
    plt.savefig(f"/home/chunxy/repos/Compass/data/distrib/{dataset}_2_{corr_rg_ub}.anticorrelated.passrate.png")
    plt.clf()
    print(f"{data[:, 1].min()}-{data[:, 1].max()} assigned to {n_queries} queries")