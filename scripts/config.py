from functools import reduce

DATASETS = [
  "sift",
  "gist",
  "glove100",
  "crawl",
  "audio",
  "video",
]
group_dataset = {
  "1": ["crawl"],
  "2": ["video", "glove100"],
  "3": ["sift", "gist", "audio"],
}

da_s = [1, 2, 3, 4]

# attribute dimension - interval, for reading JSON files
da_interval = {
  1: [
    *[((100,), (r,)) for r in (200, 300, 600)], ((0,), (10000,)),
    *[((100,), (r,)) for r in range(1100, 10000, 1000)],
  ],
  2: [
    *[((100, 200), (r1, r2)) for r1, r2 in \
      zip([1100, 3100, 5100, 8100, 9100], \
          [1200, 3200, 5200, 8200, 9200])],
  ],
  3: [
    *[((100, 200, 300), (r1, r2, r3)) for r1, r2, r3 in \
      zip([2100, 6100, 8100, 9100], \
          [2200, 6200, 8200, 9200], \
          [2300, 6300, 8300, 9300])],
  ],
  4: [
    *[((100, 200, 300, 400), (r1, r2, r3, r4)) for r1, r2, r3, r4 in \
      zip([3100, 5600, 8100, 9100], \
          [3200, 5700, 8200, 9200], \
          [3300, 5800, 8300, 9300], \
          [3400, 5900, 8400, 9400])],
  ],
}

# attribute dimension - range, for plotting
da_range = {da: ["-".join([f"{(r - l) // 100}" for l, r in zip(*itvl)]) for itvl in intervals] for da, intervals in da_interval.items()}  # noqa: E741
da_sel = {
  da:
  list(
    map(
      lambda f: f"{f:.3g}",
      map(
        lambda itvl: reduce(
          lambda a, b: a * b,
          map(
            lambda rg: (rg[1] - rg[0]) / 10000,
            zip(*itvl), ),
          1, ),
        intervals,
      ),
    )
  )
  for da, intervals in da_interval.items()
}

METHODS = [
  "CompassK",
  "CompassPca",
  "CompassBikmeans",
  "CompassKCg",
  "CompassPcaCg",
  "CompassBikmeansCg",
]

compass_parameters = {
  "build": ["M", "efc", "nlist"],
  "search": ["efs", "nrel"],
}
compass_x_parameters = {
  "build": ["M", "efc", "nlist", "dx"],
  "search": ["efs", "nrel"],
}

compass_args = {
  "M": [16, 32],
  "efc": [200],
  "nlist": [10000, 20000],
  "efs": [10, 20, 60, 100, 200, 300],
  "nrel": [100, 200],
  "dx": [64, 128, 256, 512],
}
dataset_args = {
  "sift": {
    "dx": [64],
  },
  "glove100": {
    "dx": [64],
  },
  "audio": {
    "dx": [64]
  },
  "video": {
    "dx": [64, 128, 256, 512],
  },
  "gist": {
    "dx": [64, 128, 256, 512],
  },
  "crawl": {
    "dx": [64, 128, 256]
  }
}

# method - workload template
m_workload = {
  **{
    m: "{}_10000_{}_{}_10"
    for m in METHODS
  },
}

# method - parameter
m_param = {
  "CompassK": compass_parameters,
  "CompassPca": compass_x_parameters,
  "CompassBikmeans": compass_parameters,
  "CompassKCg": compass_parameters,
  "CompassPcaCg": compass_x_parameters,
  "CompassBikmeansCg": compass_parameters,
}

m_marker = {
  "CompassK": "o",
  "CompassBikmeans": "s",
  "CompassKCg": "x",
  "CompassBikmeansCg": "d",
  "CompassPca": "o",
  "CompassPcaCg": "s",
}

b_marker = {
  "M_16_efc_200_nlist_10000": "o",
  "M_16_efc_200_nlist_20000": "s",
  "M_32_efc_200_nlist_10000": "x",
  "M_32_efc_200_nlist_20000": "d",
}
