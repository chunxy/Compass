import numpy as np

METHODS = ["Weaviate", "Milvus"]
N_QUERIES = 200
TOPK = 10

DATASET_NBASE = {
  "sift-dedup": 1000000 - 14538,
  "audio-dedup": 1000000,
  "gist-dedup": 1000000 - 17306,
  "video-dedup": 1000000,
  "glove100": 1183514,
  "crawl": 1989995,
  "flickr": 4203901,
  "deep10m": 10000000,
}

DATASET_NQUERY = {
  "sift-dedup": 10000,
  "audio-dedup": 10000,
  "gist-dedup": 1000,
  "video-dedup": 10000,
  "glove100": 10000,
  "crawl": 10000,
  "flickr": 29999,
  "deep10m": 10000,
}

DATASET_NDIM = {
  "sift-dedup": 128,
  "audio-dedup": 128,
  "gist-dedup": 960,
  "video-dedup": 1024,
  "glove100": 100,
  "crawl": 300,
  "flickr": 512,
  "deep10m": 96,
}

DATASET_M = {
  "sift-dedup": 16,
  "audio-dedup": 16,
  "gist-dedup": 16,
  "video-dedup": 32,
  "glove100": 32,
  "crawl": 16,
  "flickr": 32,
  "deep10m": 32,
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

WORKLOAD = "{}_10000_{}_{}_10"  # Search for top-10.


class Datacard:

  def __init__(
    self,
    name,
    base_path,
    query_path,
    attr_path,
    interval,
    groundtruth_path,
    workload,
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
    self.workload = workload
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
        workload=WORKLOAD.format(d, *map(lambda ele: "-".join(map(str, ele)), itvl)),
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

class RevisionDatacard:
  def __init__(
    self,
    name,
    base_path,
    query_path,
    attr_path,
    wtype,
    rg_path,
    groundtruth_path,
    workload,
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
    self.wtype = wtype
    self.rg_path = rg_path
    self.groundtruth_path = groundtruth_path
    self.workload = workload
    self.dim = dim
    self.n_base = n_base
    self.n_queries = n_queries
    self.n_groundtruth = n_groundtruth
    self.attr_dim = attr_dim


REV_ATTR = "/home/chunxy/repos/Compass/data/attr/{}_{}_{}.{}.value.bin"
REV_RG = "/home/chunxy/repos/Compass/data/range/{}_{}_{}.{}.rg.bin"  # float32
REV_GT = "/home/chunxy/repos/Compass/data/gt/{}_{}_{}.{}.hybrid.gt"  # ivecs
REV_WORKLOAD = "{}_{}_10_{}" # Search for top-10.
REV_DA_S = (1, 2, 2, 1, 1, 1)
REV_SPANS = (30, 20, 20, 30, 30, 30)
REV_WTYPES = ("skewed", "correlated", "anticorrelated", "onesided", "point", "negation")

REVISION_CARDS = {
  d: [
    RevisionDatacard(
      name=d,
      base_path=BASE.format(d, d),
      query_path=QUERY.format(d, d),
      attr_path=REV_ATTR.format(d, da, span, wtype),
      wtype=wtype,
      rg_path=REV_RG.format(d, da, span, wtype),
      groundtruth_path=REV_GT.format(d, da, span, wtype),
      workload=REV_WORKLOAD.format(d, span, wtype),
      dim=DATASET_NDIM[d],
      n_base=DATASET_NBASE[d],
      n_queries=DATASET_NQUERY[d],
      n_groundtruth=100,
      attr_dim=da,
    )
    for da, span, wtype in zip(REV_DA_S, REV_SPANS, REV_WTYPES)
  ]
  for d in DATASET_NBASE.keys()
}

REAL_DATASETS = ("flickr", "video-dedup")
RDATA_NDIM = {
  "flickr": 512,
  "video-dedup": 1024,
}
RDATA_NBASE = {
  "flickr": 4203901,
  "video-dedup": 1000000,
}
RDATA_NQUERY = {
  "flickr": 29999,
  "video-dedup": 10000,
}
REAL_DA_S = (2, 2)
REAL_SPANS = (180, 10000)
REAL_WTYPES = ("real", "real")

# Not to add REAL_CARDS first.
REAL_CARDS = {
  d: RevisionDatacard(
    name=d,
    base_path=BASE.format(d, d),
    query_path=QUERY.format(d, d),
    attr_path=REV_ATTR.format(d, da, span, wtype),
    wtype=wtype,
    rg_path=REV_RG.format(d, da, span, wtype),
    groundtruth_path=REV_GT.format(d, da, span, wtype),
    workload=REV_WORKLOAD.format(d, span, wtype),
    dim=RDATA_NDIM[d],
    n_base=RDATA_NBASE[d],
    n_queries=RDATA_NQUERY[d],
    n_groundtruth=100,
    attr_dim=da,
  )
  for d, da, span, wtype in zip(REAL_DATASETS, REAL_DA_S, REAL_SPANS, REAL_WTYPES)
}

EFS_S = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 110,
          120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260,
          270, 280, 290, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 600,
          700, 800, 900, 1000]

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

def load_float32(path, n, da):
  return np.fromfile(path, dtype=np.float32).reshape((n, da))


if __name__ == "__main__":
  print()
