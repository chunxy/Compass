from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from collections import namedtuple

import numpy as np
import json
import pandas as pd
from scipy.spatial import ConvexHull

import bisect

from itertools import product

# Directories
logs = Path("/home/chunxy/repos/Compass/logs")

# Names
ONED_METHODS = {"CompassR1d", "CompassRImi1d", "CompassIvf1d", "CompassGraph1d", "Serf"}
TWOD_METHODS = {"CompassR", "CompassIvf", "CompassGraph"}
DATASETS = {"sift", "gist", "crawl", "glove100"}
ONED_PASSRATES = {"0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "0.9"}

# Templates
METHOD_WORKLOAD_TMPL = {
  # "Compass1d": "{}_10000_{}_{}_100",
  # "Acorn": "{}_{}_100",
  "CompassR1d": "{}_10000_{}_{}_100",
  "CompassRImi1d": "{}_10000_{}_{}_100",
  "CompassIvf1d": "{}_10000_{}_{}_100",
  "CompassGraph1d": "{}_10000_{}_{}_100",
  "Serf": "{}_{}_{}_100",
  "CompassR": "{}_10000_{{{}, {}}}_{{{}, {}}}_100",
  "CompassGraph": "{}_10000_{{{}, {}}}_{{{}, {}}}_100",
  "CompassIvf": "{}_10000_{{{}, {}}}_{{{}, {}}}_100",
}

METHOD_BUILD_TMPL = {
  # "Compass1d": "M_{}_efc_{}_nlist_{}",
  # "Acorn": "M_{}_beta_{}_efc_{}_gamma_{}",
  "CompassR1d": "M_{}_efc_{}_nlist_{}",
  "CompassRImi1d": "M_{}_efc_{}_nsub_{}_nbits_{}",
  "CompassIvf1d": "nlist_{}",
  "CompassGraph1d": "M_{}_efc_{}",
  "Serf": "M_{}_efc_{}_efmax_{}",
  "CompassR": "M_{}_efc_{}_nlist_{}",
  "CompassGraph": "M_{}_efc_{}",
  "CompassIvf": "nlist_{}",
}

METHOD_SEARCH_TMPL = {
  # "Compass1d": "efs_{}_nrel_{}",
  # "Acorn": "efs_{}",
  "CompassR1d": "efs_{}_nrel_{}_mincomp_{}",
  "CompassRImi1d": "efs_{}_nrel_{}_mincomp_{}",
  "CompassIvf1d": "nprobe_{}",
  "CompassGraph1d": "efs_{}_nrel_{}",
  "Serf": "efs_{}",
  "CompassR": "efs_{}_nrel_{}_mincomp_{}",
  "CompassGraph": "efs_{}_nrel_{}",
  "CompassIvf": "nprobe_{}",
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
  RangeFilterRange(100, 9100),
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
  WindowFilterRange(100, 200, 3100, 3200),
  WindowFilterRange(100, 200, 5100, 5200),
  WindowFilterRange(100, 200, 8100, 8200),
  WindowFilterRange(100, 200, 9100, 9200),
]

SRange = namedtuple("SRange", ["range", "ndata"])
typical_sranges = [
  SRange(1000, 1_000_000),
  SRange(5000, 1_000_000),
  SRange(10000, 1_000_000),
  SRange(20000, 1_000_000),
  SRange(50000, 1_000_000),
  SRange(100000, 1_000_000),
  SRange(200000, 1_000_000),
  SRange(500000, 1_000_000),
  SRange(900000, 1_000_000),
  SRange(1900, 1_900_000),
  SRange(9500, 1_900_000),
  SRange(19000, 1_900_000),
  SRange(38000, 1_900_000),
  SRange(95000, 1_900_000),
  SRange(190000, 1_900_000),
  SRange(380000, 1_900_000),
  SRange(950000, 1_900_000),
  SRange(1710000, 1_900_000),
]

CompassBuild = namedtuple("CompassBuild", ["M", "efc", "nlist"])
typical_compass_builds = [
  CompassBuild(16, 100, 500),
  CompassBuild(16, 100, 1000),
  CompassBuild(16, 200, 500),
  CompassBuild(16, 200, 1000),
  CompassBuild(32, 100, 500),
  CompassBuild(32, 100, 1000),
  CompassBuild(32, 200, 500),
  CompassBuild(32, 200, 1000),
  CompassBuild(64, 100, 500),
  CompassBuild(64, 100, 1000),
  CompassBuild(64, 200, 500),
  CompassBuild(64, 200, 1000),
]

CompassSearch = namedtuple("CompassSearch", ["efs", "nrel", "mincomp"])
typical_compass_searches = [
  CompassSearch(100, 100, 1000),
  CompassSearch(100, 100, 2000),
  CompassSearch(100, 100, 3000),
  CompassSearch(100, 200, 1000),
  CompassSearch(100, 200, 2000),
  CompassSearch(100, 200, 3000),
  CompassSearch(200, 100, 1000),
  CompassSearch(200, 100, 2000),
  CompassSearch(200, 100, 3000),
  CompassSearch(200, 200, 1000),
  CompassSearch(200, 200, 2000),
  CompassSearch(200, 200, 3000),
  CompassSearch(300, 100, 1000),
  CompassSearch(300, 100, 2000),
  CompassSearch(300, 100, 3000),
  CompassSearch(300, 200, 1000),
  CompassSearch(300, 200, 2000),
  CompassSearch(300, 200, 3000),
]

CompassImiBuild = namedtuple("CompassImiBuild", ["M", "efc", "nsub", "nbits"])
typical_compass_imi_builds = [
  CompassImiBuild(16, 100, 4, 4),
  CompassImiBuild(16, 100, 4, 4),
  CompassImiBuild(16, 200, 4, 4),
  CompassImiBuild(16, 200, 4, 4),
  CompassImiBuild(32, 100, 4, 4),
  CompassImiBuild(32, 100, 4, 4),
  CompassImiBuild(32, 200, 4, 4),
  CompassImiBuild(32, 200, 4, 4),
  CompassImiBuild(64, 100, 4, 4),
  CompassImiBuild(64, 100, 4, 4),
  CompassImiBuild(64, 200, 4, 4),
  CompassImiBuild(64, 200, 4, 4),
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
typical_compass_searches = [
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
  SerfBuild(16, 100, 200),
  SerfBuild(16, 200, 200),
  SerfBuild(32, 100, 200),
  SerfBuild(32, 200, 200),
  SerfBuild(16, 100, 500),
  SerfBuild(16, 200, 500),
  SerfBuild(32, 100, 500),
  SerfBuild(32, 200, 500),
]

SerfSearch = namedtuple("SerfSearch", ["efs"])
typical_serf_searches = [
  SerfSearch(100),
  SerfSearch(200),
]

iRangeGraphBuild = namedtuple("iRangeGraphBuild", ["M", "efc"])
typical_i_range_graph_builds = [
  iRangeGraphBuild(16, 100),
  iRangeGraphBuild(32, 100),
]

iRangeGraphSearch = namedtuple("iRangeGraphSearch", ["efs"])
typical_i_range_graph_searches = [
  iRangeGraphSearch(10),
  iRangeGraphSearch(15),
  iRangeGraphSearch(20),
  iRangeGraphSearch(25),
  iRangeGraphSearch(30),
  iRangeGraphSearch(35),
  iRangeGraphSearch(40),
  iRangeGraphSearch(45),
  iRangeGraphSearch(50),
  iRangeGraphSearch(55),
  iRangeGraphSearch(60),
  iRangeGraphSearch(70),
  iRangeGraphSearch(80),
  iRangeGraphSearch(90),
  iRangeGraphSearch(100),
  iRangeGraphSearch(120),
  iRangeGraphSearch(140),
  iRangeGraphSearch(160),
  iRangeGraphSearch(180),
  iRangeGraphSearch(200),
  iRangeGraphSearch(250),
  iRangeGraphSearch(300),
  iRangeGraphSearch(400),
  iRangeGraphSearch(500),
  iRangeGraphSearch(600),
  iRangeGraphSearch(700),
  iRangeGraphSearch(800),
  iRangeGraphSearch(900),
  iRangeGraphSearch(1000),
  iRangeGraphSearch(1100),
  iRangeGraphSearch(1400),
  iRangeGraphSearch(1700),
]

# Mappings
METHOD_BUILD_MAPPING = {
  # "Compass1d": typical_compass_builds,
  # "Acorn": typical_acorn_builds,
  "CompassR1d": typical_compass_builds,
  "CompassRImi1d": typical_compass_imi_builds,
  "CompassIvf1d": typical_compass_ivf_builds,
  "CompassGraph1d": typical_compass_graph_builds,
  "Serf": typical_serf_builds,
  "iRangeGraph": typical_i_range_graph_builds,
  "CompassR": typical_compass_builds,
  "CompassIvf": typical_compass_ivf_builds,
  "CompassGraph": typical_compass_graph_builds,
}

METHOD_SEARCH_MAPPING = {
  # "Compass1d": typical_compass_searches,
  # "Acorn": typical_acorn_searches,
  "CompassR1d": typical_compass_searches,
  "CompassRImi1d": typical_compass_searches,
  "CompassIvf1d": typical_compass_ivf_searches,
  "CompassGraph1d": typical_compass_searches,
  "Serf": typical_serf_searches,
  "iRangeGraph": typical_i_range_graph_searches,
  "CompassR": typical_compass_searches,
  "CompassIvf": typical_compass_ivf_searches,
  "CompassGraph": typical_compass_searches,
}

METHOD_RANGE_MAPPING = {
  # "Compass1d": typical_rf_ranges,
  # "Acorn": typical_franges,
  "CompassR1d": typical_rf_ranges,
  "CompassRImi1d": typical_rf_ranges,
  "CompassIvf1d": typical_rf_ranges,
  "CompassGraph1d": typical_rf_ranges,
  "Serf": typical_sranges,
  "iRangeGraph": typical_sranges,
  "CompassR": typical_wranges,
  "CompassIvf": typical_wranges,
  "CompassGraph": typical_wranges,
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
}


def summarize_1d():
  entries = [(
    logs / m / METHOD_WORKLOAD_TMPL[m].format(d, *rg) / METHOD_BUILD_TMPL[m].format(*b) / METHOD_SEARCH_TMPL[m].format(*r),
    m,
    METHOD_WORKLOAD_TMPL[m].format(d, *rg),
    d,
    METHOD_BUILD_TMPL[m].format(*b),
    METHOD_SEARCH_TMPL[m].format(*r),
  ) for m in ONED_METHODS for d in DATASETS for rg in METHOD_RANGE_MAPPING[m] for b in METHOD_BUILD_MAPPING[m] for r in METHOD_SEARCH_MAPPING[m]]
  legal_entries = list(filter(lambda e: e[0].exists(), entries))
  df = pd.DataFrame.from_records(legal_entries, columns=["path", "method", "workload", "dataset", "build", "run"], index="path")

  rec, qps, comp, selectivities = [], [], [], []
  for e in legal_entries:
    jsons = list(e[0].glob("*.json"))
    if len(jsons) == 0:
      i = df[(df.index == e[0])].index
      df = df.drop(i)
      continue
    jsons.sort()
    with open(jsons[-1]) as f:
      stat = json.load(f)
      max_rec_prec = stat["aggregated"]["recall"]
      # if stat["aggregated"]["precision"]:
      #   max_rec_prec = max(max_rec_prec, stat["aggregated"]["precision"])
      rec.append(max_rec_prec)
      comp.append(stat["aggregated"]["num_computations"])
      qps.append(stat["aggregated"]["qps"])

    splits = e[2].split("_")
    if len(splits) == 4:
      selectivity = int(splits[1]) / int(splits[2])
    else:
      selectivity = (int(splits[3]) - int(splits[2])) / int(splits[1])
    selectivities.append(str(selectivity))

  df["recall"] = rec
  df["qps"] = qps
  df["comp"] = comp
  df["selectivity"] = selectivities
  df.to_csv("stats1d.csv")


def draw_1d_qps_recall_by_selectivity():
  types = {
    "path": str,
    "method": str,
    "workload": str,
    "build": str,
    "dataset": str,
    "selectivity": str,
    "run": str,
    "recall": float,
    "qps": float,
  }
  df = pd.read_csv("stats1d.csv", dtype=types)

  selectors = [((df["dataset"] == d) & (df["selectivity"] == r)) for d in DATASETS for r in ONED_PASSRATES]

  for selector in selectors:
    if not selector.any(): continue
    data = df[selector]
    dataset = data["dataset"].reset_index(drop=True)[0]
    selectivity = float(data["selectivity"].reset_index(drop=True)[0])

    for m in data.method.unique():
      for b in data[data["method"] == m].build.unique():
        recall_qps = data[(data["method"] == m) & (data["build"] == b)][["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
        # recall_qps = recall_qps[recall_qps["recall"] >= 0.6].to_numpy()
        recall_qps = recall_qps.to_numpy()
        plt.plot(recall_qps[:, 0], recall_qps[:, 1])
        plt.scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", marker=METHOD_MARKER_MAPPING[m])
        plt.xlabel('Recall')
        plt.ylabel('QPS')
        # plt.xticks(np.arange(0.6, 1.04, 0.05))
        plt.title("{}, Selectivity-{:.1%}".format(dataset.capitalize(), selectivity))

    plt.legend(loc="best")
    plt.savefig("figures/{}-{:.1%}-QPS-Recall.jpg".format(dataset.upper(), selectivity), dpi=200)
    plt.cla()


def draw_1d_comp_recall_by_selectivity():

  types = {
    "path": str,
    "method": str,
    "workload": str,
    "build": str,
    "dataset": str,
    "selectivity": str,
    "run": str,
    "recall": float,
    "comp": float,
  }
  df = pd.read_csv("stats1d.csv", dtype=types)

  selectors = [((df["dataset"] == d) & (df["selectivity"] == r)) for d in DATASETS for r in ONED_PASSRATES]

  for selector in selectors:
    if not selector.any(): continue
    data = df[selector]
    dataset = data["dataset"].reset_index(drop=True)[0]
    selectivity = float(data["selectivity"].reset_index(drop=True)[0])

    for m in data.method.unique():
      for b in data[data["method"] == m].build.unique():
        recall_comp = data[(data["method"] == m) & (data["build"] == b)][["recall", "comp"]].sort_values(["recall", "comp"], ascending=[True, False])
        # recall_comp = recall_comp[recall_comp["recall"] >= 0.6].to_numpy()
        recall_comp = recall_comp.to_numpy()
        plt.plot(recall_comp[:, 0], recall_comp[:, 1])
        plt.scatter(recall_comp[:, 0], recall_comp[:, 1], label=f"{m}-{b}", marker=METHOD_MARKER_MAPPING[m])
        plt.xlabel('Recall')
        plt.ylabel('# Comp')
        # plt.xticks(np.arange(0.6, 1.04, 0.05))
        plt.title("{}, Selectivity-{:.1%}".format(dataset.capitalize(), selectivity))

    plt.legend(loc="best")
    plt.savefig("figures/{}-{:.1%}-Comp-Recall.jpg".format(dataset.upper(), selectivity), dpi=200)
    plt.cla()


def draw_1d_by_dataset():
  types = {
    "path": str,
    "method": str,
    "workload": str,
    "build": str,
    "dataset": str,
    "selectivity": float,
    "run": str,
    "recall": float,
    "qps": float,
  }
  df = pd.read_csv("stats1d.csv", dtype=types)

  selectivities = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9]
  for dataset in DATASETS:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.set_yticks(np.arange(len(selectors)))
    # ax.set_yticklabels(selectivities)
    data = df[df["dataset"] == dataset]
    for m in data.method.unique():
      if m != "CompassIvf1d" and m != "CompassGraph1d": continue
      rec_qps_sel = data[data["method"] == m][["recall", "qps", "selectivity"]].sort_values(["selectivity", "recall"])
      rec_qps_sel = rec_qps_sel[rec_qps_sel["recall"] > 0.8]
      if len(rec_qps_sel) < 3: continue

      sel_s = rec_qps_sel[["selectivity"]].to_numpy().ravel()
      rec_s = rec_qps_sel[["recall"]].to_numpy().ravel()
      qps_s = rec_qps_sel[["qps"]].to_numpy().ravel()
      pos_s = np.array([bisect.bisect(selectivities, sel) for sel in sel_s])
      rec_pos = np.concatenate([rec_s[:, np.newaxis], pos_s[:, np.newaxis]], axis=1)
      order = ConvexHull(rec_pos).vertices

      pos_s, rec_s, qps_s = pos_s[order], rec_s[order], qps_s[order]
      # ax.plot_surface(
      #   recall_qps_sel[:, 0],
      #   pos,
      #   recall_qps_sel[:, 1],
      #   rstride=1,
      #   cstride=1,
      #   # facecolors=cm.rgb,
      #   linewidth=0,
      #   antialiased=False,
      #   shade=False,
      # )
      ax.plot_trisurf(pos_s, rec_s, qps_s, label=m, antialiased=True)
      # ax.plot(pos_s, rec_s, qps_s, label=m, antialiased=True, linewidth=0)
      # ax.scatter(pos_s, rec_s, qps_s, s=10, antialiased=True)
      ax.stem(pos_s, rec_s, qps_s)
      ax.set_xlabel('Selectivity')
      ax.set_ylabel('Recall')
      ax.set_zlabel('QPS')
      break
    fig.savefig("figures/{}.jpg".format(dataset), dpi=200)
    fig.clear()


def draw_1d_by_method():
  types = {
    "path": str,
    "method": str,
    "workload": str,
    "build": str,
    "dataset": str,
    "selectivity": str,
    "run": str,
    "recall": float,
    "qps": float,
  }
  df = pd.read_csv("stats1d.csv", dtype=types)

  selectivities = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "0.9"]
  for dataset in DATASETS:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_yticks(np.arange(len(selectivities)))
    ax.set_yticklabels(selectivities)
    data = df[df["dataset"] == dataset]
    for m in data.method.unique():
      rec_qps_sel = data[data["method"] == m][["recall", "qps", "selectivity"]].sort_values(["selectivity", "recall"])
      # rec_qps_sel = rec_qps_sel[rec_qps_sel["recall"] > 0.8]

      facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(selectivities)))

      def polygon_under_graph(x, y):
        """
        Construct the vertex list which defines the polygon filling the space under
        the (x, y) line graph. This assumes x is in ascending order.
        """
        return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

      verts = [
        polygon_under_graph(
          rec_qps_sel[rec_qps_sel["selectivity"] == sel]["recall"].to_numpy().ravel(),
          rec_qps_sel[rec_qps_sel["selectivity"] == sel]["qps"].to_numpy().ravel()
        ) for sel in selectivities
      ]
      pos_s = np.array(selectivities).argsort()
      poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
      ax.add_collection3d(poly, zs=range(len(selectivities)), zdir='y')

      ax.set_xlabel('Recall')
      ax.set_ylabel('Selectivity')
      ax.set_zlabel('QPS')
      fig.savefig("figures/{}-{}.jpg".format(dataset, m), dpi=200)
      fig.clear()
      break


def draw_1d_computation_by_dataset_and_selectivity():
  pass


def summarize_2d():
  entries = [(
    logs / m / METHOD_WORKLOAD_TMPL[m].format(d, *rg) / METHOD_BUILD_TMPL[m].format(*b) / METHOD_SEARCH_TMPL[m].format(*r),
    m,
    d,
    METHOD_WORKLOAD_TMPL[m].format(d, *rg),
    METHOD_BUILD_TMPL[m].format(*b),
    METHOD_SEARCH_TMPL[m].format(*r),
  ) for m in TWOD_METHODS for d in DATASETS for rg in METHOD_RANGE_MAPPING[m] for b in METHOD_BUILD_MAPPING[m] for r in METHOD_SEARCH_MAPPING[m]]
  legal_entries = list(filter(lambda e: e[0].exists(), entries))
  df = pd.DataFrame.from_records(legal_entries, columns=["path", "method", "dataset", "workload", "build", "run"], index="path")

  rec, qps, comp, selectivities = [], [], [], []
  for e in legal_entries:
    jsons = list(e[0].glob("*.json"))
    if len(jsons) == 0:
      i = df[df.index == e[0]].index
      df = df.drop(i)
      continue
    jsons.sort()
    with open(jsons[-1]) as f:
      stat = json.load(f)
      rec.append(stat["aggregated"]["recall"])
      comp.append(stat["aggregated"]["num_computations"])
      qps.append(stat["aggregated"]["qps"])

    splits = e[2].split("_")
    l_bounds, r_bounds = splits[2].strip("{}").split(", "), splits[3].strip("{}").split(", ")
    l1, l2 = list(map(int, l_bounds))
    r1, r2 = list(map(int, r_bounds))
    selectivity = (r1 - l1) * (r2 - l2) / int(splits[1]) / int(splits[1])
    selectivities.append(str(selectivity))

  df["recall"] = rec
  df["qps"] = qps
  df["comp"] = comp
  df["selectivity"] = selectivities
  df.to_csv("stats2d.csv")


def draw_2d_qps_recall_by_selectivity():
  # if not Path("stats1d.csv").exists:
  #   summarize_1d()

  types = {
    "path": str,
    "method": str,
    "workload": str,
    "build": str,
    "dataset": str,
    "selectivity": str,
    "run": str,
    "recall": float,
    "qps": float,
  }
  df = pd.read_csv("stats2d.csv", dtype=types)

  selectors = [ # grouped by selectivity
    (df["workload"] == "sift_10000_{100, 200}_{1100, 1200}_100") | (df["workload"] == "sift_100_100"),
    (df["workload"] == "sift_10000_{100, 200}_{3100, 3200}_100") | (df["workload"] == "sift_50_100"),
    (df["workload"] == "sift_10000_{100, 200}_{5100, 5200}_100") | (df["workload"] == "sift_10_100"),
    (df["workload"] == "sift_10000_{100, 200}_{8100, 8200}_100") | (df["workload"] == "sift_5_100"),
    (df["workload"] == "sift_10000_{100, 200}_{9100, 9200}_100") | (df["workload"] == "sift_2_100"),
    (df["workload"] == "gist_10000_{100, 200}_{1100, 1200}_100") | (df["workload"] == "gist_100_100"),
    (df["workload"] == "gist_10000_{100, 200}_{3100, 3200}_100") | (df["workload"] == "gist_50_100"),
    (df["workload"] == "gist_10000_{100, 200}_{5100, 5200}_100") | (df["workload"] == "gist_10_100"),
    (df["workload"] == "gist_10000_{100, 200}_{8100, 8200}_100") | (df["workload"] == "gist_5_100"),
    (df["workload"] == "gist_10000_{100, 200}_{9100, 9200}_100") | (df["workload"] == "gist_2_100"),
    (df["workload"] == "crawl_10000_{100, 200}_{1100, 1200}_100") | (df["workload"] == "crawl_100_100"),
    (df["workload"] == "crawl_10000_{100, 200}_{3100, 3200}_100") | (df["workload"] == "crawl_50_100"),
    (df["workload"] == "crawl_10000_{100, 200}_{5100, 5200}_100") | (df["workload"] == "crawl_10_100"),
    (df["workload"] == "crawl_10000_{100, 200}_{8100, 8200}_100") | (df["workload"] == "crawl_5_100"),
    (df["workload"] == "crawl_10000_{100, 200}_{9100, 9200}_100") | (df["workload"] == "crawl_2_100"),
    (df["workload"] == "glove100_10000_{100, 200}_{1100, 1200}_100") | (df["workload"] == "glove100_100_100"),
    (df["workload"] == "glove100_10000_{100, 200}_{3100, 3200}_100") | (df["workload"] == "glove100_50_100"),
    (df["workload"] == "glove100_10000_{100, 200}_{5100, 5200}_100") | (df["workload"] == "glove100_10_100"),
    (df["workload"] == "glove100_10000_{100, 200}_{8100, 8200}_100") | (df["workload"] == "glove100_5_100"),
    (df["workload"] == "glove100_10000_{100, 200}_{9100, 9200}_100") | (df["workload"] == "glove100_2_100"),
  ]

  for selector in selectors:
    if not selector.any(): continue
    data = df[selector]
    # workload = data["workload"].reset_index(drop=True)[0]
    # dataset, hrange, l, r, k = workload.split('_')
    # hrange, l, r = int(hrange), list(map(int, filter(None, re.split(r"{|}|,", l)))), list(map(int, filter(None, re.split(r"{|}|,", r))))
    # selectivity = (r[0] - l[0]) * (r[1] - l[1]) / hrange**2
    dataset = data["dataset"].reset_index(drop=True)[0]
    selectivity = data["selectivity"].reset_index(drop=True)[0]

    for m in data.method.unique():
      recall_qps = data[data["method"] == m][["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
      # recall_qps = recall_qps[recall_qps["recall"] >= 0.6].to_numpy()
      recall_qps = recall_qps.to_numpy()
      plt.plot(recall_qps[:, 0], recall_qps[:, 1])
      plt.scatter(recall_qps[:, 0], recall_qps[:, 1], label=m)
      plt.xlabel('Recall')
      plt.ylabel('QPS')
      # plt.xticks(np.arange(0.6, 1.04, 0.05))
      plt.title("{}, Selectivity-{:.1%}".format(dataset.capitalize(), selectivity))

    plt.legend(loc="best")
    plt.savefig("figures/{}2D-{:.1%}-QPS-Recall.jpg".format(dataset.upper(), selectivity), dpi=200)
    plt.cla()


def draw_2d_comp_recall_by_selectivity():
  # if not Path("stats1d.csv").exists:
  #   summarize_1d()

  types = {
    "path": str,
    "method": str,
    "workload": str,
    "build": str,
    "dataset": str,
    "selectivity": str,
    "run": str,
    "recall": float,
    "comp": float,
  }
  df = pd.read_csv("stats2d.csv", dtype=types)

  selectors = [ # grouped by selectivity
    (df["workload"] == "sift_10000_{100, 200}_{1100, 1200}_100") | (df["workload"] == "sift_100_100"),
    (df["workload"] == "sift_10000_{100, 200}_{3100, 3200}_100") | (df["workload"] == "sift_50_100"),
    (df["workload"] == "sift_10000_{100, 200}_{5100, 5200}_100") | (df["workload"] == "sift_10_100"),
    (df["workload"] == "sift_10000_{100, 200}_{8100, 8200}_100") | (df["workload"] == "sift_5_100"),
    (df["workload"] == "sift_10000_{100, 200}_{9100, 9200}_100") | (df["workload"] == "sift_2_100"),
    (df["workload"] == "gist_10000_{100, 200}_{1100, 1200}_100") | (df["workload"] == "gist_100_100"),
    (df["workload"] == "gist_10000_{100, 200}_{3100, 3200}_100") | (df["workload"] == "gist_50_100"),
    (df["workload"] == "gist_10000_{100, 200}_{5100, 5200}_100") | (df["workload"] == "gist_10_100"),
    (df["workload"] == "gist_10000_{100, 200}_{8100, 8200}_100") | (df["workload"] == "gist_5_100"),
    (df["workload"] == "gist_10000_{100, 200}_{9100, 9200}_100") | (df["workload"] == "gist_2_100"),
    (df["workload"] == "crawl_10000_{100, 200}_{1100, 1200}_100") | (df["workload"] == "crawl_100_100"),
    (df["workload"] == "crawl_10000_{100, 200}_{3100, 3200}_100") | (df["workload"] == "crawl_50_100"),
    (df["workload"] == "crawl_10000_{100, 200}_{5100, 5200}_100") | (df["workload"] == "crawl_10_100"),
    (df["workload"] == "crawl_10000_{100, 200}_{8100, 8200}_100") | (df["workload"] == "crawl_5_100"),
    (df["workload"] == "crawl_10000_{100, 200}_{9100, 9200}_100") | (df["workload"] == "crawl_2_100"),
    (df["workload"] == "glove100_10000_{100, 200}_{1100, 1200}_100") | (df["workload"] == "glove100_100_100"),
    (df["workload"] == "glove100_10000_{100, 200}_{3100, 3200}_100") | (df["workload"] == "glove100_50_100"),
    (df["workload"] == "glove100_10000_{100, 200}_{5100, 5200}_100") | (df["workload"] == "glove100_10_100"),
    (df["workload"] == "glove100_10000_{100, 200}_{8100, 8200}_100") | (df["workload"] == "glove100_5_100"),
    (df["workload"] == "glove100_10000_{100, 200}_{9100, 9200}_100") | (df["workload"] == "glove100_2_100"),
  ]

  for selector in selectors:
    if not selector.any(): continue
    data = df[selector]
    # workload = data["workload"].reset_index(drop=True)[0]
    # dataset, hrange, l, r, k = workload.split('_')
    # hrange, l, r = int(hrange), list(map(int, filter(None, re.split(r"{|}|,", l)))), list(map(int, filter(None, re.split(r"{|}|,", r))))
    # selectivity = (r[0] - l[0]) * (r[1] - l[1]) / hrange**2
    dataset = data["dataset"].reset_index(drop=True)[0]
    selectivity = data["selectivity"].reset_index(drop=True)[0]

    for m in data.method.unique():
      recall_comp = data[data["method"] == m][["recall", "comp"]].sort_values(["recall", "comp"], ascending=[True, False])
      # recall_comp = recall_comp[recall_comp["recall"] >= 0.6].to_numpy()
      recall_comp = recall_comp.to_numpy()
      plt.plot(recall_comp[:, 0], recall_comp[:, 1])
      plt.scatter(recall_comp[:, 0], recall_comp[:, 1], label=m)
      plt.xlabel('Recall')
      plt.ylabel('# Comp')
      # plt.xticks(np.arange(0.6, 1.04, 0.05))
      plt.title("{}, Selectivity-{:.1%}".format(dataset.capitalize(), selectivity))

    plt.legend(loc="best")
    plt.savefig("figures/{}2D-{:.1%}-Comp-Recall.jpg".format(dataset.upper(), selectivity), dpi=200)
    plt.cla()


plt.rcParams.update({'font.size': 15, 'legend.fontsize': 12, 'axes.labelsize': 15, 'axes.titlesize': 15, "figure.figsize": (10, 10)})
summarize_1d()
draw_1d_qps_recall_by_selectivity()
draw_1d_comp_recall_by_selectivity()
# draw_1d_ablation()
# draw_1d_by_method()

# summarize_2d()
# draw_2d_comp_recall_by_selectivity()
# draw_2d_qps_recall_by_selectivity()
