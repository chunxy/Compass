from functools import reduce

DATASETS = ["sift", "audio", "glove100", "crawl", "video", "gist"]

DA_S = [1, 2, 3, 4]

# attribute dimension - intervals, for reading JSON files of Compass result
compass_da_interval = {
  1: [
    *[((100,), (r,)) for r in (200, 300, 600)],
    *[((100,), (r,)) for r in range(1100, 10000, 1000)],
    ((0,), (10000,)),
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
sota_da_interval = {
  1: [*[(1, ), (2, ), (3, ), (5, )], *[(i, ) for i in range(10, 100, 10)]],
  2: [(pcnt, pcnt) for pcnt in (10, 30, 50, 80, 90)],
}

# attribute dimension - ranges, for plotting, shared across methods, using Compass's interval as base
DA_RANGE = {
  da: [
    "-".join([f"{(r - l) // 100}"
              for l, r in zip(*itvl)])  # noqa: E741
    for itvl in intervals
  ]
  for da, intervals in compass_da_interval.items()
}
# attribute dimension - selectivities, for plotting, shared across methods, using Compass's interval as base
DA_SEL = {
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
          1., ),
        intervals,
      ),
    )
  )
  for da, intervals in compass_da_interval.items()
}

COMPASS_METHODS = [
  "CompassK",
  "CompassBikmeans",
  "CompassKCg",
  "CompassBikmeansCg",
  "CompassPca",
  "CompassPcaCg",
]
COMPASSX_METHODS = [
  "CompassPca",
  "CompassPcaCg",
]
SOTA_METHODS = ["iRangeGraph", "SeRF"]
METHODS = COMPASS_METHODS + SOTA_METHODS

# method - workload template, accompanied with M_DA_RUN
M_WORKLOAD = {
  **{
    m: "{}_10000_{}_{}_10"
    for m in COMPASS_METHODS
  },
  **{
    m: "{}_{}_10"
    for m in SOTA_METHODS
  },
}

M_DA_RUN = {
  **{
    m: compass_da_interval
    for m in COMPASS_METHODS
  },
  **{
    m: sota_da_interval
    for m in SOTA_METHODS
  },
}

compass_group_dataset = {
  "1": ["crawl", "audio"],
  "2": ["video", "glove100"],
  "3": ["sift", "gist"],
}
compassx_group_dataset = {
  "1": ["crawl", "audio"],
  "2": ["video", "gist"],
  "3": ["sift", "glove100"],
}
M_GROUP_DATASET = {
  **{
    m: compass_group_dataset
    for m in COMPASS_METHODS
  },
  **{
    m: compassx_group_dataset
    for m in COMPASSX_METHODS
  },
}

irangegraph_parameters = {
  "build": ["M", "efc"],
  "search": ["efs"],
}
serf_parameters = {
  "build": ["M", "efc", "efmax"],
  "search": ["efs"],
}
compass_parameters = {
  "build": ["M", "efc", "nlist"],
  "search": ["efs", "nrel"],
}
compassx_parameters = {
  "build": ["M", "efc", "nlist", "dx"],
  "search": ["efs", "nrel"],
}

# method - parameter
M_PARAM = {
  "CompassK": compass_parameters,
  "CompassPca": compassx_parameters,
  "CompassBikmeans": compass_parameters,
  "CompassKCg": compass_parameters,
  "CompassPcaCg": compassx_parameters,
  "CompassBikmeansCg": compass_parameters,
  "iRangeGraph": irangegraph_parameters,
  "SeRF": serf_parameters,
}

compass_args = {
  "M": [16, 32],
  "efc": [200],
  "nlist": [10000, 20000],
  "efs": [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300],
  "nrel": [100, 200],
  "dx": [64, 128, 256, 512],
}
irangegraph_args = {
  "M": [8, 16, 32],
  "efc": [200],
  "efs": [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500],
}
serf_args = {
  "M": [16, 32],
  "efc": [200],
  "efmax": [500],
  "efs": [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200],
}
dataset_args = {
  "sift": {
    "dx": [64],
    "nlist": [5000, 10000],
  },
  "glove100": {
    "dx": [64], "efs": compass_args["efs"] + [350, 400, 500]
  },
  "audio": {
    "dx": [64],
    "nlist": [5000, 10000],
  },
  "video": {
    "dx": [256, 512],
  },
  "gist": {
    "dx": [256, 512],
  },
  "crawl": {
    "dx": [128, 256],
  }
}

M_ARGS = {
  **{
    m: compass_args
    for m in COMPASS_METHODS
  },
  "iRangeGraph": irangegraph_args,
  "SeRF": serf_args,
}

M_STYLE = {
  "CompassK": {
    "marker": "o"
  },
  "CompassPca": {
    "marker": "d"
  },
  "CompassBikmeans": {
    "marker": "s"
  },
  "CompassKCg": {
    "marker": "o", "edgecolor": "black"
  },
  "CompassPcaCg": {
    "marker": "d", "edgecolor": "black"
  },
  "CompassBikmeansCg": {
    "marker": "s", "edgecolor": "black"
  },
  "iRangeGraph": {
    "marker": "^"
  },
  "SeRF": {
    "marker": "p"
  }
}

# b_marker = {
#   "M_16_efc_200_nlist_10000": "o",
#   "M_16_efc_200_nlist_20000": "s",
#   "M_32_efc_200_nlist_10000": "x",
#   "M_32_efc_200_nlist_20000": "d",
# }
