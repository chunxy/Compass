import numpy as np

METHODS = ["Weaviate", "Milvus"]

DATASET_NBASE = {
  "sift-dedup": 1000000 - 14538,
  "audio-dedup": 1000000,
  "gist-dedup": 1000000 - 17306,
  "video-dedup": 1000000,
  "glove100": 1183514,
  "crawl": 1989995,
}

DATASET_NQUERY = {
  "sift-dedup": 10000,
  "audio-dedup": 10000,
  "gist-dedup": 1000,
  "video-dedup": 10000,
  "glove100": 10000,
  "crawl": 10000,
}

DATASET_NDIM = {
  "sift-dedup": 128,
  "audio-dedup": 128,
  "gist-dedup": 960,
  "video-dedup": 1024,
  "glove100": 100,
  "crawl": 300,
}

DATASET_M = {
  "sift-dedup": 16,
  "audio-dedup": 16,
  "gist-dedup": 16,
  "video-dedup": 32,
  "glove100": 32,
  "crawl": 16,
}

DA_S = [1, 2, 3, 4]

da_interval = {
  1: [
    *[((100,), (r,)) for r in (200, 300, 600)],
    *[((100,), (r,)) for r in range(1100, 10000, 1000)],
    ((0,), (10000,)),
  ],
  2: [
    *[((100, 200), (r1, r2)) for r1, r2 in \
      zip([1100, 1600, 2100, 2600, 3100, 3600, 4100, 4600, 5100, 5600, 6100, 6600, 7100, 7600, 8100, 8600, 9100, 9600], \
          [1200, 1700, 2200, 2700, 3200, 3700, 4200, 4700, 5200, 5700, 6200, 6700, 7200, 7700, 8200, 8700, 9200, 9700])],
  ],
  3: [
    *[((100, 200, 300), (r1, r2, r3)) for r1, r2, r3 in \
      zip([2100, 2600, 3100, 3600, 4100, 4600, 5100, 5600, 6100, 6600, 7100, 7600, 8100, 8600, 9100, 9600], \
          [2200, 2700, 3200, 3700, 4200, 4700, 5200, 5700, 6200, 6700, 7200, 7700, 8200, 8700, 9200, 9700], \
          [2300, 2800, 3300, 3800, 4300, 4800, 5300, 5800, 6300, 6800, 7300, 7800, 8300, 8800, 9300, 9800])],
  ],
  4: [
    *[((100, 200, 300, 400), (r1, r2, r3, r4)) for r1, r2, r3, r4 in \
      zip([2100, 2600, 3100, 3600, 4100, 4600, 5100, 5600, 6100, 6600, 7100, 7600, 8100, 8600, 9100, 9600], \
          [2200, 2700, 3200, 3700, 4200, 4700, 5200, 5700, 6200, 6700, 7200, 7700, 8200, 8700, 9200, 9700], \
          [2300, 2800, 3300, 3800, 4300, 4800, 5300, 5800, 6300, 6800, 7300, 7800, 8300, 8800, 9300, 9800], \
          [2400, 2900, 3400, 3900, 4400, 4900, 5400, 5900, 6400, 6900, 7400, 7900, 8400, 8900, 9400, 9900])],
  ],
}

WORKLOAD = "{}_10000_{}_{}_10"
# w = WORKLOAD.format(d, *map(lambda ele: "-".join(map(str, ele)), itvl))


class Datacard:

  def __init__(
    self,
    name,
    base_path,
    query_path,
    attr_path,
    interval,
    groundtruth_path,
    dim,
    n_base,
    n_queries,
    n_groundtruth,
    attr_dim,
  ):
    self.name = name
    self.base_path = base_path
    self.query_path = query_path
    self.attr_path = attr_path
    self.interval = interval
    self.groundtruth_path = groundtruth_path
    self.dim = dim
    self.n_base = n_base
    self.n_queries = n_queries
    self.n_groundtruth = n_groundtruth
    self.attr_dim = attr_dim


BASE = "/home/chunxy/datasets/{}/{}_base.fvecs"
QUERY = "/home/chunxy/datasets/{}/{}_query.fvecs"
ATTR = "/home/chunxy/repos/Compass/data/attr/{}_{}_{}.value.bin"
GT = "/home/chunxy/repos/Compass/data/gt/{}_{}_{{{}}}_{{{}}}_{}.hybrid.gt"

CARDS = {
  d: {
    da: [
      Datacard(
        name=d,
        base_path=BASE.format(d, d),
        query_path=QUERY.format(d, d),
        attr_path=ATTR.format(d, da, 10000),
        interval=itvl,
        groundtruth_path=GT.format(d, 10000, ", ".join(map(str, itvl[0])), ", ".join(map(str, itvl[1])), 100),
        dim=DATASET_NDIM[d],
        n_base=DATASET_NBASE[d],
        n_queries=DATASET_NQUERY[d],
        n_groundtruth=100,
        attr_dim=da,
      ) for itvl in da_interval[da]
    ]
    for da in DA_S
  }
  for d in DATASET_NBASE.keys()
}


def load_fvecs(path, n, dim):
  fvecs = np.fromfile(path, dtype=np.float32).reshape((n, dim + 1))
  data = fvecs[:, 1:]
  return data


def load_ivecs(path, n, dim):
  ivecs = np.fromfile(path, dtype=np.uint32).reshape((n, dim + 1))
  data = ivecs[:, 1:]
  return data


def load_attr(path, n, da):
  attrs = np.fromfile(path, dtype=np.float32).reshape((n, da + 1))
  data = attrs[:, 1:]
  return data


if __name__ == "__main__":
  print()
