from pathlib import Path
from collections import namedtuple
from itertools import product



# Directories
logs = Path("/home/chunxy/repos/Compass/logs")

# Names
ONED_METHODS = {"CompassR1d", "CompassRImi1d", "CompassIvf1d", "CompassGraph1d", "Serf", "iRangeGraph"}
TWOD_METHODS = {"CompassR", "CompassIvf", "CompassGraph", "iRangeGraph2d"}
DATASETS = {"sift", "gist", "crawl", "glove100", "audio", "video"}
ONED_PASSRATES = {"0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"}
TWOD_PASSRATES = {
  "0.01",
  "0.02",
  "0.05",
  "0.1",
  "0.2",
}

# Templates
METHOD_WORKLOAD_TMPL = {
  # "Compass1d": "{}_10000_{}_{}_100",
  # "Acorn": "{}_{}_100",
  "CompassR1d": "{}_10000_{}_{}_100",
  "CompassRImi1d": "{}_10000_{}_{}_100",
  "CompassIvf1d": "{}_10000_{}_{}_100",
  "CompassGraph1d": "{}_10000_{}_{}_100",
  "Serf": "{}_{}_{}_100",
  "iRangeGraph": "{}_{}_{}_100",
  "CompassR": "{}_10000_{{{}, {}}}_{{{}, {}}}_100",
  "CompassGraph": "{}_10000_{{{}, {}}}_{{{}, {}}}_100",
  "CompassIvf": "{}_10000_{{{}, {}}}_{{{}, {}}}_100",
  "iRangeGraph2d": "{}_{}_{}_100",
}

METHOD_BUILD_TMPL = {
  # "Compass1d": "M_{}_efc_{}_nlist_{}",
  # "Acorn": "M_{}_beta_{}_efc_{}_gamma_{}",
  "CompassR1d": "M_{}_efc_{}_nlist_{}",
  "CompassRImi1d": "M_{}_efc_{}_nsub_{}_nbits_{}",
  "CompassIvf1d": "nlist_{}",
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
  "CompassRImi1d": "efs_{}_nrel_{}_mincomp_{}",
  "CompassIvf1d": "nprobe_{}",
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
  RangeFilterRange(100, 200),
  RangeFilterRange(100, 300),
  RangeFilterRange(100, 600),
  RangeFilterRange(100, 1100),
  RangeFilterRange(100, 2100),
  RangeFilterRange(100, 5100),
  RangeFilterRange(100, 6100),
  RangeFilterRange(100, 7100),
  RangeFilterRange(100, 8100),
  RangeFilterRange(100, 9100),
  RangeFilterRange(0, 10000),
]

LabelFilterRange = namedtuple("LabelFilterRange", ["max"])
typical_franges = [
  LabelFilterRange(2),
  LabelFilterRange(5),
  LabelFilterRange(10),
  LabelFilterRange(50),
  LabelFilterRange(100),
  LabelFilterRange(500),
  LabelFilterRange(1000),
]

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
  FractionRange(1000, 1_000_000),
  FractionRange(5000, 1_000_000),
  FractionRange(10000, 1_000_000),
  FractionRange(20000, 1_000_000),
  FractionRange(50000, 1_000_000),
  FractionRange(100000, 1_000_000),
  FractionRange(200000, 1_000_000),
  FractionRange(500000, 1_000_000),
  FractionRange(600000, 1_000_000),
  FractionRange(700000, 1_000_000),
  FractionRange(800000, 1_000_000),
  FractionRange(900000, 1_000_000),
  FractionRange(1000000, 1_000_000),
  # FractionRange(1900, 1_900_000),
  # FractionRange(9500, 1_900_000),
  FractionRange(19_000, 1_900_000),
  FractionRange(38_000, 1_900_000),
  FractionRange(95_000, 1_900_000),
  FractionRange(190_000, 1_900_000),
  FractionRange(380_000, 1_900_000),
  FractionRange(950_000, 1_900_000),
  FractionRange(1140_000, 1_900_000),
  FractionRange(1330_000, 1_900_000),
  FractionRange(1520_000, 1_900_000),
  FractionRange(1710_000, 1_900_000),
  FractionRange(1900_000, 1_900_000),
]

typical_irange_graph_2d_ranges = [
  FractionRange(10000, 1_000_000),
  FractionRange(20000, 1_000_000),
  FractionRange(50000, 1_000_000),
  FractionRange(100000, 1_000_000),
  FractionRange(200000, 1_000_000),
  FractionRange(19000, 1_900_000),
  FractionRange(38000, 1_900_000),
  FractionRange(95000, 1_900_000),
  FractionRange(190000, 1_900_000),
  FractionRange(380000, 1_900_000),
]

CompassBuild = namedtuple("CompassBuild", ["M", "efc", "nlist"])
typical_compass_1d_builds = [
  # CompassBuild(16, 100, 500),
  # CompassBuild(16, 100, 1000),
  # CompassBuild(16, 200, 500),
  # CompassBuild(16, 200, 1000),
  # CompassBuild(32, 100, 500),
  CompassBuild(32, 100, 1000),
  # CompassBuild(32, 200, 500),
  CompassBuild(32, 200, 1000),
  # CompassBuild(64, 100, 500),
  CompassBuild(64, 100, 1000),
  # CompassBuild(64, 200, 500),
  CompassBuild(64, 200, 1000),
]

CompassSearch = namedtuple("CompassSearch", ["efs", "nrel", "mincomp"])
typical_compass_1d_searches = [
  CompassSearch(100, 500, 1000),
  CompassSearch(120, 500, 1000),
  CompassSearch(140, 500, 1000),
  CompassSearch(150, 500, 1000),
  CompassSearch(160, 500, 1000),
  CompassSearch(180, 500, 1000),
  CompassSearch(200, 500, 1000),
  CompassSearch(250, 500, 1000),
]

typical_compass_2d_searches = [
  CompassSearch(100, 100, 1000),
  CompassSearch(250, 100, 1000),
]

CompassImiBuild = namedtuple("CompassImiBuild", ["M", "efc", "nsub", "nbits"])
typical_compass_imi_builds = [
  CompassImiBuild(16, 100, 4, 4),
  CompassImiBuild(16, 200, 4, 4),
  CompassImiBuild(32, 100, 4, 4),
  CompassImiBuild(32, 200, 4, 4),
  CompassImiBuild(64, 100, 4, 4),
  CompassImiBuild(64, 200, 4, 4),
  CompassImiBuild(16, 100, 2, 9),
  CompassImiBuild(16, 100, 2, 9),
  CompassImiBuild(16, 200, 2, 9),
  CompassImiBuild(16, 200, 2, 9),
  CompassImiBuild(32, 100, 2, 9),
  CompassImiBuild(32, 200, 2, 9),
  CompassImiBuild(64, 100, 2, 9),
  CompassImiBuild(64, 200, 2, 9),
]

CompassIvfBuild = namedtuple("CompassIvfBuild", ["nlist"])
typical_compass_ivf_builds = [
  CompassIvfBuild(100),
  CompassIvfBuild(500),
  CompassIvfBuild(1000),
]

CompassIvfSearch = namedtuple("CompassIvfSearch", ["nprobe"])
typical_compass_ivf_searches = [
  CompassIvfSearch(10),
  CompassIvfSearch(100),
  CompassIvfSearch(200),
  CompassIvfSearch(500),
  CompassIvfSearch(1000),
]

CompassGraphBuild = namedtuple("CompassGraphBuild", ["M", "efc"])
typical_compass_graph_builds = [
  CompassGraphBuild(16, 100),
  CompassGraphBuild(16, 200),
  CompassGraphBuild(32, 100),
  CompassGraphBuild(32, 200),
  CompassGraphBuild(64, 100),
  CompassGraphBuild(64, 200),
]

CompassGraphSearch = namedtuple("CompassGraphSearch", ["efs", "nrel"])
typical_compass_graph_searches = [
  CompassGraphSearch(100, 100),
  CompassGraphSearch(100, 200),
  CompassGraphSearch(200, 100),
  CompassGraphSearch(200, 200),
  CompassGraphSearch(300, 100),
  CompassGraphSearch(300, 200),
]

# AcornBuild = namedtuple("AcornBuild", ["M", "beta", "efc", "gamma"])
# typical_acorn_builds = [AcornBuild(*build) for build in product(M_s, beta_s, efc_s, gamma_s)]

# AcornSearch = namedtuple("AcornSearch", ["efs"])
# typical_acorn_searches = [AcornSearch(search) for search in product(efs_s)]

SerfBuild = namedtuple("SerfBuild", ["M", "efc", "efmax"])
typical_serf_builds = [
  # SerfBuild(16, 100, 200),
  # SerfBuild(16, 200, 200),
  # SerfBuild(32, 100, 200),
  # SerfBuild(32, 200, 200),
  # SerfBuild(64, 100, 200),
  # SerfBuild(64, 200, 200),
  # SerfBuild(16, 100, 500),
  # SerfBuild(16, 200, 500),
  # SerfBuild(32, 100, 500),
  SerfBuild(32, 200, 500),
  # SerfBuild(64, 100, 500),
  # SerfBuild(64, 200, 500),
]

SerfSearch = namedtuple("SerfSearch", ["efs"])
typical_serf_searches = [
  SerfSearch(100),
  SerfSearch(120),
  SerfSearch(140),
  SerfSearch(150),
  SerfSearch(160),
  SerfSearch(180),
  SerfSearch(200),
]

iRangeGraphBuild = namedtuple("iRangeGraphBuild", ["M", "efc"])
typical_i_range_graph_builds = [
  iRangeGraphBuild(16, 100),
  iRangeGraphBuild(16, 200),
  iRangeGraphBuild(32, 100),
  iRangeGraphBuild(32, 200),
]

iRangeGraphSearch = namedtuple("iRangeGraphSearch", ["efs"])
typical_i_range_graph_searches = [
  iRangeGraphSearch(100),
  iRangeGraphSearch(120),
  iRangeGraphSearch(140),
  iRangeGraphSearch(160),
  iRangeGraphSearch(180),
  iRangeGraphSearch(200),
  iRangeGraphSearch(250),
  iRangeGraphSearch(300),
  # iRangeGraphSearch(400),
  # iRangeGraphSearch(500),
]

# Mappings
METHOD_BUILD_MAPPING = {
  # "Compass1d": typical_compass_builds,
  # "Acorn": typical_acorn_builds,
  "CompassR1d": typical_compass_1d_builds,
  "CompassRImi1d": typical_compass_imi_builds,
  "CompassIvf1d": typical_compass_ivf_builds,
  "CompassGraph1d": typical_compass_graph_builds,
  "Serf": typical_serf_builds,
  "iRangeGraph": typical_i_range_graph_builds,
  "CompassR": typical_compass_1d_builds,
  "CompassIvf": typical_compass_ivf_builds,
  "CompassGraph": typical_compass_graph_builds,
  "iRangeGraph2d": typical_i_range_graph_builds,
}

METHOD_SEARCH_MAPPING = {
  # "Compass1d": typical_compass_searches,
  # "Acorn": typical_acorn_searches,
  "CompassR1d": typical_compass_1d_searches,
  "CompassRImi1d": typical_compass_1d_searches,
  "CompassIvf1d": typical_compass_ivf_searches,
  "CompassGraph1d": typical_compass_1d_searches,
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
  "CompassRImi1d": typical_rf_ranges,
  "CompassIvf1d": typical_rf_ranges,
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
  "CompassRImi1d": 'v',
  "CompassIvf1d": '*',
  "CompassGraph1d": '^',
  "Serf": ',',
  "iRangeGraph": '2',
  "CompassR": 'o',
  "CompassIvf": '*',
  "CompassGraph": '^',
  "iRangeGraph2d": '2',
}