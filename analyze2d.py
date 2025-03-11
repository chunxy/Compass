from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import json
import pandas as pd
from scipy.spatial import ConvexHull

from .config import *


def summarize_2d():
  entries = [(
    logs_100 / m / METHOD_WORKLOAD_TMPL[m].format(d, *rg) / METHOD_BUILD_TMPL[m].format(*b) / METHOD_SEARCH_TMPL[m].format(*r),
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

    splits = e[3].split("_")
    if len(splits) == 5:
      l_bounds, r_bounds = splits[2].strip("{}").split(", "), splits[3].strip("{}").split(", ")
      l1, l2 = list(map(int, l_bounds))
      r1, r2 = list(map(int, r_bounds))
      selectivity = (r1 - l1) * (r2 - l2) / int(splits[1]) / int(splits[1])
    else:
      selectivity = int(splits[1]) / int(splits[2])
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

  selectors = [((df["dataset"] == d) & (df["selectivity"] == r)) for d in DATASETS for r in TWOD_PASSRATES]

  for selector in selectors:
    if not selector.any(): continue
    data = df[selector]
    # workload = data["workload"].reset_index(drop=True)[0]
    # dataset, hrange, l, r, k = workload.split('_')
    # hrange, l, r = int(hrange), list(map(int, filter(None, re.split(r"{|}|,", l)))), list(map(int, filter(None, re.split(r"{|}|,", r))))
    # selectivity = (r[0] - l[0]) * (r[1] - l[1]) / hrange**2
    dataset = data["dataset"].reset_index(drop=True)[0]
    selectivity = float(data["selectivity"].reset_index(drop=True)[0])

    plt.subplots(layout='constrained')
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

    plt.gcf().legend(loc="outside right upper")
    plt.savefig("figures2d/{}-{:.1%}-QPS-Recall.jpg".format(dataset.upper(), selectivity), dpi=200)
    plt.close()


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

  selectors = [((df["dataset"] == d) & (df["selectivity"] == r)) for d in DATASETS for r in TWOD_PASSRATES]

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

    plt.gcf().legend(loc="outside right upper")
    plt.savefig("figures2d/{}-{:.1%}-Comp-Recall.jpg".format(dataset.upper(), selectivity), dpi=200)
    plt.close()


# summarize_2d()
# draw_2d_comp_recall_by_selectivity()
# draw_2d_qps_recall_by_selectivity()
