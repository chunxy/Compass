from pathlib import Path
from collections import namedtuple
from itertools import product

# Directories
LOGS_TMPL = "/home/chunxy/repos/Compass/logs_{}"

# Names
ONED_METHODS = {"CompassR1d", "CompassROld1d", "CompassRImi1d", "CompassIvf1d", "CompassImi1d", "CompassGraph1d", "Serf", "iRangeGraph"}
TWOD_METHODS = {"CompassR", "CompassIvf", "CompassGraph", "iRangeGraph2d"}
DATASETS = {"sift", "gist", "crawl", "glove100", "audio", "video"}
ONED_PASSRATES = {"0.01", "0.02", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"}
TWOD_PASSRATES = {"0.01", "0.02", "0.05", "0.1", "0.2"}

# Templates
METHOD_WORKLOAD_TMPL = {
  # "Compass1d": "{}_10000_{}_{}_{}",
  # "Acorn": "{}_{}_{}",
  "CompassR1d": "{}_10000_{}_{}_{}",
  "CompassROld1d": "{}_10000_{}_{}_{}",
  "CompassRImi1d": "{}_10000_{}_{}_{}",
  "CompassIvf1d": "{}_10000_{}_{}_{}",
  "CompassImi1d": "{}_10000_{}_{}_{}",
  "CompassGraph1d": "{}_10000_{}_{}_{}",
  "Serf": "{}_{}_{}_{}",
  "iRangeGraph": "{}_{}_{}_{}",
  "CompassR": "{}_10000_{{{}, {}}}_{{{}, {}}}_{}",
  "CompassGraph": "{}_10000_{{{}, {}}}_{{{}, {}}}_{}",
  "CompassIvf": "{}_10000_{{{}, {}}}_{{{}, {}}}_{}",
  "iRangeGraph2d": "{}_{}_{}_{}",
}

METHOD_BUILD_TMPL = {
  # "Compass1d": "M_{}_efc_{}_nlist_{}",
  # "Acorn": "M_{}_beta_{}_efc_{}_gamma_{}",
  "CompassR1d": "M_{}_efc_{}_nlist_{}",
  "CompassROld1d": "M_{}_efc_{}_nlist_{}",
  "CompassRImi1d": "M_{}_efc_{}_nsub_{}_nbits_{}",
  "CompassIvf1d": "nlist_{}",
  "CompassImi1d": "nsub_{}_nbits_{}",
  "CompassGraph1d": "M_{}_efc_{}",
  "Serf": "M_{}_efc_{}_efmax_{}",
  "iRangeGraph": "M_{}_efc_{}",
  "CompassR": "M_{}_efc_{}_nlist_{}",
  "CompassGraph": "M_{}_efc_{}",
  "CompassIvf": "nlist_{}",
  "iRangeGraph2d": "M_{}_efc_{}",
}

METHOD_SEARCH_TMPL = {
  # "Compass1d": "efs_{}_nrel_{}",
  # "Acorn": "efs_{}",
  "CompassR1d": "efs_{}_nrel_{}_mincomp_{}",
  "CompassROld1d": "efs_{}_nrel_{}_mincomp_{}",
  "CompassRImi1d": "efs_{}_nrel_{}_mincomp_{}",
  "CompassIvf1d": "nprobe_{}",
  "CompassImi1d": "nprobe_{}",
  "CompassGraph1d": "efs_{}_nrel_{}",
  "Serf": "efs_{}",
  "iRangeGraph": "efs_{}",
  "CompassR": "efs_{}_nrel_{}_mincomp_{}",
  "CompassGraph": "efs_{}_nrel_{}",
  "CompassIvf": "nprobe_{}",
  "iRangeGraph2d": "efs_{}",
}

METHOD_PARAM_TMPL = {m: METHOD_BUILD_TMPL[m] + '_' + METHOD_SEARCH_TMPL[m] for m in ONED_METHODS | TWOD_METHODS}

# Arguments
RangeFilterRange = namedtuple("RangeFilterRange", ["l", "r"])
typical_rf_ranges = [
  *[RangeFilterRange(100, r) for r in (200, 300, 600, 1100, 2100, 3100, 4100, 5100, 6100, 7100, 8100, 9100)],
  RangeFilterRange(0, 10000),
]
typical_ivf_rf_ranges = [*[RangeFilterRange(100, r) for r in (200, 300, 600, 1100)]]

LabelFilterRange = namedtuple("LabelFilterRange", ["max"])
typical_franges = [*[LabelFilterRange(n) for n in (2, 5, 10, 50, 100, 500, 1000)]]

WindowFilterRange = namedtuple("WindowFilterRange", ["l1", "l2", "r1", "r2"])
typical_wranges = [
  WindowFilterRange(100, 200, 1100, 1200),
  WindowFilterRange(100, 200, 1100, 2200),
  WindowFilterRange(100, 200, 1100, 5200),
  WindowFilterRange(100, 200, 2100, 5200),
  WindowFilterRange(100, 200, 4100, 5200),
]

FractionRange = namedtuple("FractionRange", ["range", "ndata"])
typical_serf_2d_ranges = [
  *[
    FractionRange(r, 1_000_000) for r in (
      # 1000,
      # 5000,
      10000,
      20000,
      50000,
      100000,
      200000,
      300000,
      400000,
      500000,
      600000,
      700000,
      800000,
      900000,
      1000000,
    )
  ],
  *[
    FractionRange(r, 1_900_000) for r in (
      # 1900,
      # 9500,
      19_000,
      38_000,
      95_000,
      190_000,
      380_000,
      570_000,
      760_000,
      950_000,
      1140_000,
      1330_000,
      1520_000,
      1710_000,
      1900_000,
    )
  ]
]

typical_irange_graph_2d_ranges = [
  *[FractionRange(r, 1_000_000) for r in (10000, 20000, 50000, 100000, 200000)],
  *[FractionRange(r, 1_900_000) for r in (19_000, 38_000, 95_000, 190_000, 380_000)],
]

CompassBuild = namedtuple("CompassBuild", ["M", "efc", "nlist"])
typical_compass_r_old_1d_builds = [
  CompassBuild(16, 200, 1000),
  CompassBuild(32, 200, 1000),
  CompassBuild(32, 200, 2000),  # *[CompassBuild(M, efc, nlist) for M, efc, nlist in product([16, 32, 64], [100, 200], [500, 100])]
]

typical_compass_r_1d_builds = [
  CompassBuild(16, 200, 1000),
  CompassBuild(32, 200, 1000),  # *[CompassBuild(M, efc, nlist) for M, efc, nlist in product([16, 32, 64], [100, 200], [500, 100])]
]

CompassSearch = namedtuple("CompassSearch", ["efs", "nrel", "mincomp"])
typical_compass_1d_searches = [
  # *[CompassSearch(efs, 500, 1000) for efs in (100, 110, 120, 130, 140, 150, 160, 180, 200, 250, 300)],
  *[CompassSearch(efs, 500, 1000) for efs in (10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200)],
  *[CompassSearch(efs, nrel, 1000) for efs, nrel in product([20, 40, 60, 100, 200], [500, 1000, 1500, 2000, 3000, 3500, 4000])]
]
typical_compass_1d_old_searches = [
  *[CompassSearch(efs, nrel, 1000) for efs, nrel in product([10, 20, 25, 30, 35, 40, 50, 60, 100, 200], [500, 600, 700, 800, 1000, 1500])],
  *[CompassSearch(efs, nrel, 1000) for efs, nrel in product([300, 500], [100, 500, 1000, 1500])]
]
typical_compass_2d_searches = [CompassSearch(100, 100, 1000), CompassSearch(250, 100, 1000)]

CompassRImiBuild = namedtuple("CompassRImiBuild", ["M", "efc", "nsub", "nbits"])
typical_compass_r_imi_builds = [*[CompassRImiBuild(M, efc, *imi) for M, efc, imi in product([16, 32, 64], [100, 200], [(4, 4), (2, 9)])]]

CompassIvfBuild = namedtuple("CompassIvfBuild", ["nlist"])
typical_compass_ivf_builds = [*[CompassIvfBuild(nlist) for nlist in (500, 1000)]]

CompassIvfSearch = namedtuple("CompassIvfSearch", ["nprobe"])
typical_compass_ivf_searches = [*[CompassIvfSearch(nprobe) for nprobe in (10, 20, 30, 40, 50, 100)]]

CompassImiBuild = namedtuple("CompassImiBuild", ["nsub", "nbits"])
typical_compass_imi_builds = [CompassImiBuild(2, 8)]

CompassImiSearch = namedtuple("CompassImiSearch", ["nprobe"])
typical_compass_imi_searches = [*[CompassImiSearch(nprobe) for nprobe in (100, 150, 200, 250, 300, 400, 500)]]

CompassGraphBuild = namedtuple("CompassGraphBuild", ["M", "efc"])
typical_compass_graph_builds = [*[CompassGraphBuild(M, efc) for M, efc in product([16, 32], [100, 200])]]

CompassGraphSearch = namedtuple("CompassGraphSearch", ["efs", "nrel"])
typical_compass_graph_1d_searches = [
  *[
    CompassGraphSearch(efs, nrel) for efs, nrel in product([10, 20, 60, 100, 200], [500, 1000, 1500, 2000, 3000, 3500, 4000, 5000, 6000, 7000, 8000])
  ]
]

# AcornBuild = namedtuple("AcornBuild", ["M", "beta", "efc", "gamma"])
# typical_acorn_builds = [AcornBuild(*build) for build in product(M_s, beta_s, efc_s, gamma_s)]

# AcornSearch = namedtuple("AcornSearch", ["efs"])
# typical_acorn_searches = [AcornSearch(search) for search in product(efs_s)]

SerfBuild = namedtuple("SerfBuild", ["M", "efc", "efmax"])
typical_serf_builds = [
  SerfBuild(16, 200, 500),
  SerfBuild(32, 200, 500),  # *[SerfBuild(M, efc, efmax) for M, efc, efmax in product([16, 32, 64], [100, 200], [200, 500])],
]

SerfSearch = namedtuple("SerfSearch", ["efs"])
typical_serf_searches = [
  # *[SerfSearch(efs) for efs in (100, 110, 120, 130, 140, 150, 160, 180, 200, 250, 300)],
  *[SerfSearch(efs) for efs in (10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200)],
]

iRangeGraphBuild = namedtuple("iRangeGraphBuild", ["M", "efc"])
typical_i_range_graph_builds = [
  # iRangeGraphBuild(16, 100),
  iRangeGraphBuild(16, 200),
  # iRangeGraphBuild(32, 100),
  iRangeGraphBuild(32, 200),
]

iRangeGraphSearch = namedtuple("iRangeGraphSearch", ["efs"])
typical_i_range_graph_searches = [
  # *[iRangeGraphSearch(efs) for efs in (100, 110, 120, 130, 140, 150, 160, 180, 200, 250, 300)],
  *[iRangeGraphSearch(efs) for efs in (10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200)],
]

# Mappings
METHOD_BUILD_MAPPING = {
  # "Compass1d": typical_compass_builds,
  # "Acorn": typical_acorn_builds,
  "CompassR1d": typical_compass_r_1d_builds,
  "CompassROld1d": typical_compass_r_old_1d_builds,
  "CompassRImi1d": typical_compass_r_imi_builds,
  "CompassIvf1d": typical_compass_ivf_builds,
  "CompassImi1d": typical_compass_imi_builds,
  "CompassGraph1d": typical_compass_graph_builds,
  "Serf": typical_serf_builds,
  "iRangeGraph": typical_i_range_graph_builds,
  "CompassR": typical_compass_r_1d_builds,
  "CompassIvf": typical_compass_ivf_builds,
  "CompassGraph": typical_compass_graph_builds,
  "iRangeGraph2d": typical_i_range_graph_builds,
}

METHOD_SEARCH_MAPPING = {
  # "Compass1d": typical_compass_searches,
  # "Acorn": typical_acorn_searches,
  "CompassR1d": typical_compass_1d_searches,
  "CompassROld1d": typical_compass_1d_old_searches,
  "CompassRImi1d": typical_compass_1d_searches,
  "CompassIvf1d": typical_compass_ivf_searches,
  "CompassImi1d": typical_compass_imi_searches,
  "CompassGraph1d": typical_compass_graph_1d_searches,
  "Serf": typical_serf_searches,
  "iRangeGraph": typical_i_range_graph_searches,
  "CompassR": typical_compass_2d_searches,
  "CompassIvf": typical_compass_ivf_searches,
  "CompassGraph": typical_compass_1d_searches,
  "iRangeGraph2d": typical_i_range_graph_searches,
}

METHOD_RANGE_MAPPING = {
  # "Compass1d": typical_rf_ranges,
  # "Acorn": typical_franges,
  "CompassR1d": typical_rf_ranges,
  "CompassROld1d": typical_rf_ranges,
  "CompassRImi1d": typical_rf_ranges,
  "CompassIvf1d": typical_ivf_rf_ranges,
  "CompassImi1d": typical_ivf_rf_ranges,
  "CompassGraph1d": typical_rf_ranges,
  "Serf": typical_serf_2d_ranges,
  "iRangeGraph": typical_serf_2d_ranges,
  "CompassR": typical_wranges,
  "CompassIvf": typical_wranges,
  "CompassGraph": typical_wranges,
  "iRangeGraph2d": typical_irange_graph_2d_ranges,
}

METHOD_MARKER_MAPPING = {
  # "Compass1d": typical_rf_ranges,
  "CompassR1d": 'o',
  "CompassROld1d": 'p',
  "CompassRImi1d": 'v',
  "CompassIvf1d": '*',
  "CompassImi1d": 'X',
  "CompassGraph1d": '^',
  "Serf": ',',
  "iRangeGraph": '2',
  "CompassR": 'o',
  "CompassIvf": '*',
  "CompassGraph": '^',
  "iRangeGraph2d": '2',
}

COMPASS_BUILD_MARKER_MAPPING = {
  "M_16_efc_200_nlist_1000": "D",
  "M_32_efc_200_nlist_1000": "p",
  "M_32_efc_200_nlist_2000": "|",
}
