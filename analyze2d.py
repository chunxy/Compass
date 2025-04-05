from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import json
import pandas as pd
from scipy.spatial import ConvexHull

from .config import *

K = 100
LOGS = Path(LOGS_TMPL.format(K))

def summarize_2d():
  entries = [(
    LOGS / m / TEMPLATES[m].workload.format(d, *rg, K) / TEMPLATES[m].build.format(*b) / TEMPLATES[m].search.format(*r),
    m,
    d,
    TEMPLATES[m].workload.format(d, *rg, K),
    TEMPLATES[m].build.format(*b),
    TEMPLATES[m].search.format(*r),
  ) for m in TWOD_METHODS for d in DATASETS for rg in TWOD_RUNS[m].range for b in TWOD_RUNS[m].build for r in TWOD_RUNS[m].search]
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
  df.to_csv(f"stats2d_{K}.csv")


def draw_2d_qps_comp_wrt_recall_by_dataset_selectivity():
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
  df = pd.read_csv(f"stats2d_{K}.csv", dtype=types)
  selectors = [((df["dataset"] == d) & (df["selectivity"] == r)) for d in DATASETS for r in TWOD_PASSRATES]

  for selector in selectors:
    if not selector.any(): continue
    data = df[selector]
    dataset = data["dataset"].reset_index(drop=True)[0]
    selectivity = float(data["selectivity"].reset_index(drop=True)[0])

    fig, axs = plt.subplots(2, 1, layout='constrained')
    for m in data.method.unique():
      for b in data[data["method"] == m].build.unique():
        data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
        recall_qps = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
        recall_qps = recall_qps.to_numpy()
        axs[0].plot(recall_qps[:, 0], recall_qps[:, 1])
        axs[0].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", marker=TWOD_RUNS[m].marker)
        axs[0].xlabel('Recall')
        axs[0].ylabel('QPS')
        axs[0].title(f"{dataset.upper()}, Selectivity-{selectivity:.1%}")

        comp_qps = data_by_m_b[["comp", "qps"]].sort_values(["comp", "qps"], ascending=[True, True])
        comp_qps = comp_qps.to_numpy()
        axs[1].plot(comp_qps[:, 0], comp_qps[:, 1])
        axs[1].scatter(comp_qps[:, 0], comp_qps[:, 1], label=f"{m}-{b}", marker=TWOD_RUNS[m].marker)
        axs[1].xlabel('Recall')
        axs[1].ylabel('# Comp')
        axs[1].title(f"{dataset.upper()}, Selectivity-{selectivity:.1%}")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper")
    fig.savefig(f"figures2d_{K}/{dataset.upper()}-{selectivity:.1%}-QPS-Recall.jpg", dpi=200)
    plt.close()

# summarize_2d()
# draw_2d_qps_comp_wrt_recall_by_dataset_selectivity()

