from collections import namedtuple
from itertools import product

# This file is mainly for summarizing JSON files into CSV file.
# For figures, configs are set up separately.

# Directories
LOGS_TMPL = "/home/chunxy/repos/Compass/logs_{}"

# Names
ONED_METHODS = ("CompassRImi1d", "CompassIvf1d", "CompassImi1d", "CompassGraph1d", "Serf", "iRangeGraph")
ONED_METHODS += ("CompassRR1dBikmeans", "CompassRRCg1dBikmeans", "CompassRRCg1dPca")
TWOD_METHODS = ("CompassRRBikmeans", "CompassRRCgBikmeans", "CompassIvf", "CompassGraph", "iRangeGraph2d")
DATASETS = ("sift", "audio", "video", "crawl", "gist", "glove100")
ONED_PASSRATES = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
# TWOD_RANGES = [f"{pcnt1}-{pcnt2}" for pcnt1, pcnt2 in product([1, 5, 10, 30, 50, 80, 90], [1, 5, 10, 30, 50, 80, 90])]
TWOD_RANGES = [f"{pcnt}-{pcnt}" for pcnt in [1, 5, 10, 30, 50, 80, 90]]

# Arguments
RangeFilterRange = namedtuple("RangeFilterRange", ["l", "r"])
typical_rf_ranges = [
  *[RangeFilterRange(100, r) for r in (200, 300, 600, 1100, 2100, 3100, 4100, 5100, 6100, 7100, 8100, 9100)],
  RangeFilterRange(0, 10000),
]
typical_graph_ranges = [
  *[RangeFilterRange(100, r) for r in (600, 1100, 2100, 3100, 4100, 5100, 6100, 7100, 8100, 9100)],
  RangeFilterRange(0, 10000),
]
typical_ivf_rf_ranges = [*[RangeFilterRange(100, r) for r in (200, 300, 600, 1100, 2100)]]

# LabelFilterRange = namedtuple("LabelFilterRange", ["max"])
# typical_franges = [*[LabelFilterRange(n) for n in (2, 5, 10, 50, 100, 500, 1000)]]

WindowFilterRange = namedtuple("WindowFilterRange", ["l1", "l2", "r1", "r2"])
typical_wf_ranges = [
  *[WindowFilterRange(100, 200, r1, r2) for r1, r2 in product([200, 600, 1100, 3100, 5100, 8100, 9100], [300, 700, 1200, 3200, 5200, 8200, 9200])]
  # WindowFilterRange(100, 200, 1100, 2200),
  # WindowFilterRange(100, 200, 1100, 5200),
  # WindowFilterRange(100, 200, 2100, 5200),
  # WindowFilterRange(100, 200, 4100, 5200),
]
typical_ivf_wf_ranges = [
  WindowFilterRange(100, 200, 1100, 1200),
  WindowFilterRange(100, 200, 3100, 3200),
  WindowFilterRange(100, 200, 5100, 5200),
]
typical_graph_wf_ranges = [
  WindowFilterRange(100, 200, 3100, 3200),
  WindowFilterRange(100, 200, 5100, 5200),
  WindowFilterRange(100, 200, 8100, 8200),
  WindowFilterRange(100, 200, 9100, 9200),
]

FractionRange = namedtuple("FractionRange", ["range", "ndata"])
typical_fraction_ranges = [
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
      1000000, )
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
      1900_000, )
  ]
]

WindowFilterPercentageRange = namedtuple("WindowFilterPercentageRange", ["pcnt1", "pcnt2"])
typical_irangegraph_2d_ranges = [
  *[WindowFilterPercentageRange(pcnt1, pcnt2) for pcnt1, pcnt2 in product([1, 5, 10, 30, 50, 80, 90], [1, 5, 10, 30, 50, 80, 90])],
]

CompassBuild = namedtuple("CompassBuild", ["M", "efc", "nlist"])
typical_compass_r_old_1d_builds = [
  CompassBuild(16, 200, 1000),
  CompassBuild(16, 200, 2000),
  CompassBuild(32, 200, 1000),
  CompassBuild(32, 200, 2000),
  CompassBuild(32, 200, 5000),
  CompassBuild(32, 200, 10000),
]
typical_compass_r_1d_builds = [
  CompassBuild(16, 200, 1000),
  CompassBuild(16, 200, 2000),
  CompassBuild(16, 200, 5000),
  CompassBuild(16, 200, 10000),
  CompassBuild(16, 200, 20000),
  CompassBuild(32, 200, 1000),
  CompassBuild(32, 200, 2000),
  CompassBuild(32, 200, 5000),
  CompassBuild(32, 200, 10000),
  CompassBuild(32, 200, 20000),
]
typical_compass_r_cg_1d_builds = [
  CompassBuild(16, 200, 1000),
  CompassBuild(16, 200, 2000),
  CompassBuild(16, 200, 5000),
  CompassBuild(16, 200, 10000),
  CompassBuild(16, 200, 20000),
  CompassBuild(32, 200, 1000),
  CompassBuild(32, 200, 2000),
  CompassBuild(32, 200, 5000),
  CompassBuild(32, 200, 10000),
  CompassBuild(32, 200, 20000),
]
typical_compass_r_builds = typical_compass_r_1d_builds
typical_compass_r_cg_builds = typical_compass_r_cg_1d_builds

CompassSearch = namedtuple("CompassSearch", ["efs", "nrel", "mincomp"])
typical_compass_r_1d_searches = [
  # *[CompassSearch(efs, 500, 1000) for efs in (100, 110, 120, 130, 140, 150, 160, 180, 200, 250, 300)],
  *[CompassSearch(efs, nrel, 1000) for efs, nrel in product([10, 20, 60, 100, 200], [500, 600, 800, 1000])],
  # *[CompassSearch(efs, nrel, 1000) for efs, nrel in product([10, 20, 60, 100, 120, 140, 160, 180, 200, 250, 300, 400, 500], [100, 200])]
]
typical_compass_r_cg_1d_searches = [
  *[CompassSearch(efs, nrel, 1000) for efs in (10, 20, 60, 100, 200) for nrel in (100, 200)],
]
typical_compass_r_old_1d_searches = [
  *[CompassSearch(efs, nrel, 1000) for efs, nrel in product([10, 15, 20, 25, 30, 35, 40, 50, 60, 100, 200], [500, 600, 700, 800, 1000, 1500])],
  *[CompassSearch(efs, nrel, 1000) for efs, nrel in product([300, 500], [100, 500, 1000, 1500])]
]
typical_compass_r_searches = [CompassSearch(efs, nrel, 1000) for efs, nrel in product([10, 20, 60, 100, 200], [500, 600, 800, 1000])]
typical_compass_r_cg_searches = [CompassSearch(efs, nrel, 1000) for efs, nrel in product([10, 20, 60, 100, 200], [500, 600, 800, 1000])]

CompassRImiBuild = namedtuple("CompassRImiBuild", ["M", "efc", "nsub", "nbits"])
typical_compass_r_imi_builds = [*[CompassRImiBuild(M, efc, *imi) for M, efc, imi in product([16, 32, 64], [100, 200], [(4, 4), (2, 9)])]]

CompassIvfBuild = namedtuple("CompassIvfBuild", ["nlist"])
typical_compass_ivf_1d_builds = [*[CompassIvfBuild(nlist) for nlist in (1000, 2000, 5000, 10000)]]
typical_compass_ivf_builds = [*[CompassIvfBuild(nlist) for nlist in (1000, 2000, 5000, 10000)]]

CompassIvfSearch = namedtuple("CompassIvfSearch", ["nprobe"])
typical_compass_ivf_1d_searches = [
  *[CompassIvfSearch(nprobe) for nprobe in (10, 15, 20, 25, 30, 35, 40, 45, 50, 100)], *[CompassIvfSearch(nprobe) for nprobe in (60, 80, 150)]
]
typical_compass_ivf_searches = [*[CompassIvfSearch(nprobe) for nprobe in (10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100)]]

CompassImiBuild = namedtuple("CompassImiBuild", ["nsub", "nbits"])
typical_compass_imi_1d_builds = [CompassImiBuild(2, 8)]

CompassImiSearch = namedtuple("CompassImiSearch", ["nprobe"])
typical_compass_imi_1d_searches = [*[CompassImiSearch(nprobe) for nprobe in (100, 150, 200, 250, 300, 400, 500)]]

CompassGraphBuild = namedtuple("CompassGraphBuild", ["M", "efc"])
typical_compass_graph_1d_builds = [*[CompassGraphBuild(M, 200) for M in [16, 32]]]
typical_compass_graph_builds = typical_compass_graph_1d_builds

CompassGraphSearch = namedtuple("CompassGraphSearch", ["efs", "nrel"])
typical_compass_graph_1d_searches = [
  *[CompassGraphSearch(efs, nrel) for efs, nrel in product([10, 20, 60, 100, 200, 300, 400, 500, 1000, 1500, 2000], [100, 200])]
]
typical_compass_graph_searches = [
  *[CompassGraphSearch(efs, nrel) for efs, nrel in product([10, 20, 60, 100, 200, 300, 400, 500, 1000, 1500, 2000], [100, 200])],
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
typical_irangegraph_builds = [
  iRangeGraphBuild(8, 200),
  iRangeGraphBuild(16, 200),
  iRangeGraphBuild(32, 200),
]

iRangeGraphSearch = namedtuple("iRangeGraphSearch", ["efs"])
typical_irangegraph_searches = [
  # *[iRangeGraphSearch(efs) for efs in (100, 110, 120, 130, 140, 150, 160, 180, 200, 250, 300)],
  *[iRangeGraphSearch(efs) for efs in (10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500)],
]

# Runs
Run = namedtuple("Run", ["name", "range", "build", "search", "marker"])
ONED_RUNS = {
  "CompassR1d": Run("CompassR1d", typical_rf_ranges, typical_compass_r_1d_builds, typical_compass_r_1d_searches, "o"),
  # "CompassROld1d": Run("CompassROld1d", typical_rf_ranges, typical_compass_r_old_1d_builds, typical_compass_r_old_1d_searches, "p"),
  # "CompassRImi1d": Run("CompassRImim1d", typical_rf_ranges, typical_compass_r_imi_builds, typical_compass_r_1d_searches, "v"),
  # "CompassRCg1d": Run("CompassRCg1d", typical_rf_ranges, typical_compass_r_1d_builds, typical_compass_r_cg_1d_searches, "<"),
  # "CompassRR1d": Run("CompassRR1d", typical_rf_ranges, typical_compass_r_1d_builds, typical_compass_r_cg_1d_searches, "<"),
  "CompassRR1dBikmeans": Run("CompassRR1dBikmeans", typical_rf_ranges, typical_compass_r_1d_builds, typical_compass_r_cg_1d_searches, "<"),
  # "CompassRR1dKmedoids": Run("CompassRR1dKmedoids", typical_rf_ranges, typical_compass_r_1d_builds, typical_compass_r_cg_1d_searches, "<"),
  # "CompassRRCg1d": Run("CompassRRCg1d", typical_rf_ranges, typical_compass_r_1d_builds, typical_compass_r_cg_1d_searches, "<"),
  "CompassRRCg1dBikmeans": Run("CompassRRCg1dBikmeans", typical_rf_ranges, typical_compass_r_1d_builds, typical_compass_r_cg_1d_searches, "<"),
  "CompassRRCg1dPca": Run("CompassRRCg1dPca", typical_rf_ranges, typical_compass_r_1d_builds, typical_compass_r_cg_1d_searches, "^"),
  # "CompassRRCg1dKmedoids": Run("CompassRRCg1dKmedoids", typical_rf_ranges, typical_compass_r_1d_builds, typical_compass_r_cg_1d_searches, "<"),
  "CompassGraph1d": Run("CompassGraph1d", typical_graph_ranges, typical_compass_graph_1d_builds, typical_compass_graph_1d_searches, "^"),
  "CompassIvf1d": Run("CompassIvf1d", typical_ivf_rf_ranges, typical_compass_ivf_1d_builds, typical_compass_ivf_1d_searches, "*"),
  # "CompassImi1d": Run("CompassImi1d", typical_ivf_rf_ranges, typical_compass_imi_1d_builds, typical_compass_imi_1d_searches, "v"),
  "Serf": Run("Serf", typical_fraction_ranges, typical_serf_builds, typical_serf_searches, ","),
  "iRangeGraph": Run("iRangeGraph", typical_fraction_ranges, typical_irangegraph_builds, typical_irangegraph_searches, "2"),
}
TWOD_RUNS = {
  "CompassRRBikmeans": Run("CompassRRBikmeans", typical_wf_ranges, typical_compass_r_builds, typical_compass_r_searches, "o"),
  "CompassRRCgBikmeans": Run("CompassRRCgBikmeans", typical_wf_ranges, typical_compass_r_cg_builds, typical_compass_r_cg_searches, "<"),
  "CompassIvf": Run("CompassIvf", typical_ivf_wf_ranges, typical_compass_ivf_builds, typical_compass_ivf_searches, "*"),
  "CompassGraph": Run("CompassGraph", typical_graph_wf_ranges, typical_compass_graph_builds, typical_compass_graph_searches, "^"),
  "iRangeGraph2d": Run("iRangeGraph2d", typical_irangegraph_2d_ranges, typical_irangegraph_builds, typical_irangegraph_searches, "2"),
}

# Templates
Template = namedtuple("Template", ["workload", "build", "search"])
TEMPLATES = {
  # "Acorn": Template("{}_{}_{}", "M_{}_beta_{}_efc_{}_gamma_{}", "efs_{}"),
  "CompassR1d": Template("{}_10000_{}_{}_{}", "M_{}_efc_{}_nlist_{}", "efs_{}_nrel_{}_mincomp_{}"),
  "CompassROld1d": Template("{}_10000_{}_{}_{}", "M_{}_efc_{}_nlist_{}", "efs_{}_nrel_{}_mincomp_{}"),
  "CompassRImi1d": Template("{}_10000_{}_{}_{}", "M_{}_efc_{}_nsub_{}_nbits_{}", "efs_{}_nrel_{}_mincomp_{}"),
  "CompassRCg1d": Template("{}_10000_{}_{}_{}", "M_{}_efc_{}_nlist_{}", "efs_{}_nrel_{}_mincomp_{}"),
  "CompassRR1d": Template("{}_10000_{}_{}_{}", "M_{}_efc_{}_nlist_{}", "efs_{}_nrel_{}_mincomp_{}"),
  "CompassRR1dBikmeans": Template("{}_10000_{}_{}_{}", "M_{}_efc_{}_nlist_{}", "efs_{}_nrel_{}_mincomp_{}"),
  "CompassRR1dKmedoids": Template("{}_10000_{}_{}_{}", "M_{}_efc_{}_nlist_{}", "efs_{}_nrel_{}_mincomp_{}"),
  "CompassRRCg1d": Template("{}_10000_{}_{}_{}", "M_{}_efc_{}_nlist_{}", "efs_{}_nrel_{}_mincomp_{}"),
  "CompassRRCg1dBikmeans": Template("{}_10000_{}_{}_{}", "M_{}_efc_{}_nlist_{}", "efs_{}_nrel_{}_mincomp_{}"),
  "CompassRRCg1dKmedoids": Template("{}_10000_{}_{}_{}", "M_{}_efc_{}_nlist_{}", "efs_{}_nrel_{}_mincomp_{}"),
  "CompassIvf1d": Template("{}_10000_{}_{}_{}", "nlist_{}", "nprobe_{}"),
  "CompassImi1d": Template("{}_10000_{}_{}_{}", "nsub_{}_nbits_{}", "nprobe_{}"),
  "CompassGraph1d": Template("{}_10000_{}_{}_{}", "M_{}_efc_{}", "efs_{}_nrel_{}"),
  "Serf": Template("{}_{}_{}_{}", "M_{}_efc_{}_efmax_{}", "efs_{}"),
  "iRangeGraph": Template("{}_{}_{}_{}", "M_{}_efc_{}", "efs_{}"),
  "CompassRRBikmeans": Template("{}_10000_{{{}, {}}}_{{{}, {}}}_{}", "M_{}_efc_{}_nlist_{}", "efs_{}_nrel_{}"),
  "CompassRRCgBikmeans": Template("{}_10000_{{{}, {}}}_{{{}, {}}}_{}", "M_{}_efc_{}_nlist_{}", "efs_{}_nrel_{}"),
  "CompassGraph": Template("{}_10000_{{{}, {}}}_{{{}, {}}}_{}", "M_{}_efc_{}", "efs_{}_nrel_{}"),
  "CompassIvf": Template("{}_10000_{{{}, {}}}_{{{}, {}}}_{}", "nlist_{}", "nprobe_{}"),
  "iRangeGraph2d": Template("{}_{}_{}_{}", "M_{}_efc_{}", "efs_{}"),
}

COMPASS_BUILD_MARKER_MAPPING = {
  "M_16_efc_200_nlist_1000": "D",
  "M_16_efc_200_nlist_5000": "h",
  "M_16_efc_200_nlist_10000": "p",
  "M_32_efc_200_nlist_1000": "8",
  "M_32_efc_200_nlist_5000": ">",
  "M_32_efc_200_nlist_10000": "P",
}
