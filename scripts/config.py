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
      zip([3100, 3600, 4100, 4600, 5100, 5600, 6100, 6600, 7100, 7600, 8100, 8600, 9100, 9600], \
          [3200, 3700, 4200, 4700, 5200, 5700, 6200, 6700, 7200, 7700, 8200, 8700, 9200, 9700], \
          [3300, 3800, 4300, 4800, 5300, 5800, 6300, 6800, 7300, 7800, 8300, 8800, 9300, 9800], \
          [3400, 3900, 4400, 4900, 5400, 5900, 6400, 6900, 7400, 7900, 8400, 8900, 9400, 9900])],
  ],
}
sota_da_interval = {
  1: [*[(1, ), (2, ), (3, ), (5, )], *[(i, ) for i in range(10, 100, 10)]],
  2: [(pcnt, pcnt) for pcnt in (10, 20, 30, 40, 50, 60, 70, 80, 90)],
}
post_da_interval = {
  2: [(pcnt, pcnt) for pcnt in (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95)],
  3: [(pcnt, pcnt, pcnt) for pcnt in (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95)],
  4: [(pcnt, pcnt, pcnt, pcnt) for pcnt in (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95)],
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
  "CompassKIcg",
  "CompassBikmeansIcg",
  "CompassPcaIcg",
  "CompassKQicg",
  "CompassBikmeansQicg",
  "CompassPcaQicg",
]
COMPASSX_METHODS = [
  "CompassPca",
  "CompassPcaCg",
  "CompassPcaIcg",
  "CompassPcaQicg",
]
SOTA_METHODS = ["iRangeGraph", "SeRF"]
METHODS = COMPASS_METHODS + SOTA_METHODS
POSTFILTERING_METHODS = ["SeRF+Post", "iRangeGraph+Post"]

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
  **{
    m: "{}_{}_10"
    for m in POSTFILTERING_METHODS
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
  ** {
    m: post_da_interval
    for m in POSTFILTERING_METHODS
  }
}

compass_group_dataset = {
  "1": ["crawl", "audio"],
  "2": ["video", "glove100"],
  "3": ["sift", "gist"],
}
compassx_group_dataset = {
  "1": ["crawl", "audio"],
  "2": ["video", "glove100"],
  "3": ["sift", "gist"],
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
compass_cg_parameters = {
  "build": ["M", "efc", "nlist", "M_cg"],
  "search": ["efs", "nrel"],
}
compass_icg_parameters = {
  "build": ["M", "efc", "nlist", "M_cg"],
  "search": ["efs", "nrel", "batch_k", "initial_efs", "delta_efs"],
}
compass_x_parameters = {
  "build": ["M", "efc", "nlist", "dx"],
  "search": ["efs", "nrel"],
}
compass_x_cg_parameters = {
  "build": ["M", "efc", "nlist", "dx", "M_cg"],
  "search": ["efs", "nrel"],
}
compass_x_icg_parameters = {
  "build": ["M", "efc", "nlist", "dx", "M_cg"],
  "search": ["efs", "nrel", "batch_k", "initial_efs", "delta_efs"],
}

# method - parameter
M_PARAM = {
  "CompassK": compass_parameters,
  "CompassBikmeans": compass_parameters,
  "CompassKCg": compass_cg_parameters,
  "CompassBikmeansCg": compass_cg_parameters,
  "CompassKIcg": compass_icg_parameters,
  "CompassBikmeansIcg": compass_icg_parameters,
  "CompassKQicg": compass_icg_parameters,
  "CompassBikmeansQicg": compass_icg_parameters,
  "CompassPca": compass_x_parameters,
  "CompassPcaCg": compass_x_cg_parameters,
  "CompassPcaIcg": compass_x_icg_parameters,
  "CompassPcaQicg": compass_x_icg_parameters,
  "iRangeGraph": irangegraph_parameters,
  "SeRF": serf_parameters,
  "iRangeGraph+Post": irangegraph_parameters,
  "SeRF+Post": serf_parameters,
}

compass_args = {
  "M": [16, 32],
  "efc": [200],
  "nlist": [1000, 2000, 5000, 10000, 20000],
  "efs": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200, 220, 240, 260, 280, 300],
  "nrel": [50, 100],
  "dx": [64, 128, 256, 512],
  "M_cg": [4],
  "batch_k": [50],
  "initial_efs": [50],
  "delta_efs": [30],  # 100 for old exps, # 50 for old iterative graph
}
irangegraph_args = {
  "M": [16, 32],
  "efc": [200],
  "efs": [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500],
}
serf_args = {
  "M": [16, 32],
  "efc": [200],
  "efmax": [500],
  "efs": [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 1000],
}
MULTIPLES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
M_ARGS = {
  **{
    m: compass_args
    for m in COMPASS_METHODS
  },
  "iRangeGraph": irangegraph_args,
  "SeRF": serf_args,
  "iRangeGraph+Post": irangegraph_args,
  "SeRF+Post": serf_args,
}

D_ARGS = {
  "sift": {
    "dx": [64],
    "nlist": [1000, 2000, 5000, 10000],
  },
  "glove100": {
    "dx": [64],
    "efs": compass_args["efs"] + [250, 350, 400, 450, 500, 600, 700, 800, 900, 1000],
  },
  "audio": {
    "dx": [64],
    "nlist": [1000, 2000, 5000, 10000],
  },
  "video": {
    "dx": [256, 512],
    "efs": compass_args["efs"] + [250, 350, 400, 450, 500, 600, 700, 800, 900, 1000],
  },
  "gist": {
    "dx": [256, 512],
    "efs": compass_args["efs"] + [250, 350, 400, 450, 500],
  },
  "crawl": {
    "dx": [128, 256],
    # "M_cg": [4, 8],
  }
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
  "CompassKIcg": {
    "marker": "o", "edgecolor": "red"
  },
  "CompassPcaIcg": {
    "marker": "d", "edgecolor": "red"
  },
  "CompassBikmeansIcg": {
    "marker": "s", "edgecolor": "red"
  },
  "CompassKQicg": {
    "marker": "o", "edgecolor": "yellow"
  },
  "CompassPcaQicg": {
    "marker": "d", "edgecolor": "yellow"
  },
  "CompassBikmeansQicg": {
    "marker": "s", "edgecolor": "yellow"
  },
  "iRangeGraph": {
    "marker": "^", "color": "black"
  },
  "SeRF": {
    "marker": "p", "color": "gray"
  },
  "iRangeGraph+Post": {
    "marker": "^", "color": "black"
  },
  "SeRF+Post": {
    "marker": "p", "color": "gray"
  }
}
