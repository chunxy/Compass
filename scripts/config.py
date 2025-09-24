from functools import reduce

DATASETS = ["sift", "audio", "crawl", "gist", "video", "glove100"]

DA_S = [1]

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
sota_post_da_interval = {
  2: [(pcnt, pcnt) for pcnt in (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95)],
  3: [(pcnt, pcnt, pcnt) for pcnt in (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95)],
  4: [(pcnt, pcnt, pcnt, pcnt) for pcnt in (10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95)],
}
ivf_da_interval = {
  1: [*[((100, ), (r, )) for r in (200, 300, 600, 1100, 2100)]]
}
compass_graph_da_interval = {
  1: [*[((100, ), (r, )) for r in range(2100, 10000, 1000)]],
}
postfiltering_da_interval = {
  1: [
    # *[((100, ), (r, )) for r in (200, 300, 600)], # temporarily removed to avoid figure scaling
    *[((100, ), (r, )) for r in range(1100, 10000, 1000)],
  ]
}
prefiltering_da_interval = {
  1: [*[((100, ), (r, )) for r in (200, 300, 600)]]
}
navix_da_interval = {
  # 1: [*[((0, ), (r * 1_000_000 // 100, )) for r in (1, 3, 5, 10, 20, 30, 40, 50, 75, 90)]] # temporarily removed to avoid figure scaling
  1: [*[((0, ), (r * 1_000_000 // 100, )) for r in (5, 10, 20, 30, 40, 50, 60, 70, 75, 80, 90)]]
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
  # "CompassKCg",
  # "CompassBikmeansCg",
  "CompassPca",
  # "CompassPcaCg",
  "CompassKIcg",
  "CompassBikmeansIcg",
  "CompassPcaIcg",
  # "CompassKQicg",
  # "CompassBikmeansQicg",
  # "CompassPcaQicg",
]
COMPASSX_METHODS = [
  "CompassPca",
  # "CompassPcaCg",
  "CompassPcaIcg",
  # "CompassPcaQicg",
]
SOTA_METHODS = ["iRangeGraph", "SeRF"]
BASE_METHODS = ["Prefiltering", "Postfiltering", "CompassPostK", "CompassPostKTh", "Ivf", "CompassGraph", "Navix"]
METHODS = COMPASS_METHODS + SOTA_METHODS + BASE_METHODS
SOTA_POST_METHODS = ["SeRF+Post", "iRangeGraph+Post"]

# method - workload template, accompanied with M_DA_RUN
M_WORKLOAD = {
  **{
    m: "{}_10000_{}_{}_10"
    for m in COMPASS_METHODS + BASE_METHODS
  },
  **{
    m: "{}_{}_10"
    for m in SOTA_METHODS
  },
  **{
    m: "{}_{}_10"
    for m in SOTA_POST_METHODS
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
  **{
    m: sota_post_da_interval
    for m in SOTA_POST_METHODS
  },
  "Prefiltering": prefiltering_da_interval,
  "Postfiltering": postfiltering_da_interval,
  "CompassPostK": compass_da_interval,
  "CompassPostKTh": compass_da_interval,
  "Ivf": ivf_da_interval,
  "CompassGraph": compass_graph_da_interval,
  "Navix": navix_da_interval,
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
compass_post_group_dataset = {
  "1": ["crawl"],
  "2": ["audio"],
  "3": ["video"],
  "4": ["glove100"],
  "5": ["sift"],
  "6": ["gist"],
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
  **{
    m: compass_group_dataset
    for m in BASE_METHODS
  },
  "CompassPostK": compass_post_group_dataset,
  "CompassPostKTh": compass_post_group_dataset,
  "Ivf": compass_post_group_dataset,
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
compass_post_parameters = {
  "build": ["M", "efc", "nlist", "M_cg"],
  "search": ["efs", "nrel", "batch_k", "initial_efs", "delta_efs"],
}
ivf_parameters = {
  "build": ["nlist"],
  "search": ["nprobe"],
}
prefiltering_parameters = {"build": [], "search": []}
postfiltering_parameters = {"build": ["M", "efc"], "search": ["efs"]}
compass_graph_parameters = {"build": ["M", "efc"], "search": ["efs", "nrel"]}
navix_parameters = {"build": ["M", "efc"], "search": ["efs"]}

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
  "Prefiltering": prefiltering_parameters,
  "Postfiltering": postfiltering_parameters,
  "CompassPostK": compass_post_parameters,
  "CompassPostKTh": compass_post_parameters,
  "Ivf": ivf_parameters,
  "CompassGraph": compass_graph_parameters,
  "Navix": navix_parameters,
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
  "delta_efs": [30, 31],  # 100 for old exps, 50 for old iterative graph, 31 for grouped attributes
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
prefiltering_args = {}
postfiltering_args = {
  "M": [16, 32],
  "efc": [200],
  "efs": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000],
}
compass_post_args = {
  "M": [16, 32],
  "efc": [200],
  "nlist": [10000, 20000],
  "efs": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
  "M_cg": [4],
  "nrel": [50, 100],
  "batch_k": [50],
  "initial_efs": [50],
  "delta_efs": [30],  # 100 for old exps, 50 for old iterative graph, 31 for grouped attributes
}
compass_post_th_args = {
  "M": [16, 32],
  "efc": [200],
  "nlist": [10000],
  "efs": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
  "M_cg": [4],
  "nrel": [50, 100],
  # "batch_k": [50],
  "batch_k": [20],
  # "initial_efs": [50],
  "initial_efs": [20],
  "delta_efs": [20],
}
ivf_args = {
  "nlist": [10000, 20000],
  "nprobe": [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 160, 180, 200],
}
compass_graph_args = {
  "M": [16, 32],
  "efc": [200],
  "efs": [10, 20, 60, 100, 200, 300, 400, 500, 1000, 1500, 2000],
  "nrel": [100, 200],
}
navix_args = {
  "M": [16],
  "efc": [200],
  "efs": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
}
M_ARGS = {
  **{
    m: compass_args
    for m in COMPASS_METHODS
  },
  "iRangeGraph": irangegraph_args,
  "SeRF": serf_args,
  "iRangeGraph+Post": irangegraph_args,
  "SeRF+Post": serf_args,
  "Prefiltering": prefiltering_args,
  "Postfiltering": postfiltering_args,
  "CompassPostK": compass_post_args,
  "CompassPostKTh": compass_post_th_args,
  "Ivf": ivf_args,
  "CompassGraph": compass_graph_args,
  "Navix": navix_args,
}

D_ARGS = {
  "sift": {
    "dx": [64],
    "nlist": [5000, 10000],
  },
  "glove100": {
    "dx": [64],
    "efs": compass_args["efs"] + [210, 220, 230, 240, 250, 260, 270, 280, 290, 300],
  },
  "audio": {
    "dx": [64],
    "nlist": [5000, 10000],
  },
  "video": {
    "dx": [256, 512],
    "nlist": [10000, 20000],
    "efs": compass_args["efs"] + [210, 220, 230, 240, 250, 260, 270, 280, 290, 300],
  },
  "gist": {
    "dx": [256, 512],
    "efs": compass_args["efs"] + [210, 220, 230, 240, 250, 260, 270, 280, 290, 300],
  },
  "crawl": {
    "dx": [128, 256],
    "nlist": [10000, 20000],
    "M_cg": [8],
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
  },
  "Prefiltering": {
    "marker": "^", "color": "red"
  },
  "Postfiltering": {
    "marker": "p", "color": "green"
  },
  "CompassPostK": {
    "marker": "s", "color": "blue"
  },
  "CompassPostKTh": {
    "marker": "o", "color": "red"
  },
  "Ivf": {
    "marker": "^", "color": "pink"
  },
  "CompassGraph": {
    "marker": "^", "color": "orange"
  },
  "Navix": {
    "marker": "^", "color": "purple"
  },
}
