from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import bisect
from collections import OrderedDict

from config import *

K = 10
LOGS = Path(LOGS_TMPL.format(K))


def draw_2d_comp_wrt_recall_by_selectivity(selected_workloads, selected_builds, selected_searches=None, subdir=""):
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

  selectors = [df["selectivity"] == r for r in selected_workloads]

  for selector in selectors:
    if not selector.any(): continue
    data = df[selector]
    selectivity = data["selectivity"].reset_index(drop=True)[0]

    nrow = 2
    ncol = (len(DATASETS) + nrow - 1) // nrow
    fig, axs = plt.subplots(nrow, ncol, layout='constrained')
    for i, dataset in enumerate(DATASETS):
      for m in selected_builds:
        for b in [TEMPLATES[m].build.format(*_) for _ in selected_builds[m]]:
          marker = COMPASS_BUILD_MARKER_MAPPING.get(b, TWOD_RUNS[m].marker)
          data_by_m_b = data[(data["method"] == m) & (data["build"] == b) & (data["dataset"] == dataset)]
          if selected_searches and m in selected_searches:
            for s in selected_searches[m]:
              data_by_m_b_s = data_by_m_b[data_by_m_b["run"].str.contains(s)]
              if data_by_m_b_s.size == 0: continue
              comp_recall = data_by_m_b_s[["recall", "comp"]].sort_values(["recall", "comp"], ascending=[True, True])
              comp_recall = comp_recall.to_numpy()
              axs[i // ncol, i % ncol].plot(comp_recall[:, 0], comp_recall[:, 1])
              axs[i // ncol, i % ncol].scatter(comp_recall[:, 0], comp_recall[:, 1], label=f"{m}-{b}-{s}", marker=marker)
          else:
            comp_recall = data_by_m_b[["recall", "comp"]].sort_values(["recall", "comp"], ascending=[True, True])
            comp_recall = comp_recall.to_numpy()
            axs[i // ncol, i % ncol].plot(comp_recall[:, 0], comp_recall[:, 1])
            axs[i // ncol, i % ncol].scatter(comp_recall[:, 0], comp_recall[:, 1], label=f"{m}-{b}", marker=marker)

          axs[i // ncol, i % ncol].set_xlabel('Recall')
          axs[i // ncol, i % ncol].set_ylabel('# Comp')
          axs[i // ncol, i % ncol].set_title("{}, Selectivity-{}".format(dataset.capitalize(), selectivity))

    fig.set_size_inches(15, 10)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside right upper')
    fig.savefig(f"cherrypick2d_{K}/" + f"{subdir}" + f"All-{selectivity}-Comp-Recall.jpg", dpi=200)
    plt.close()


def draw_2d_qps_wrt_recall_by_selectivity(selected_workloads, selected_builds, selected_searches=None, markers=None, subdir=""):
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

  selectors = [df["selectivity"] == r for r in selected_workloads]

  for selector in selectors:
    if not selector.any(): continue
    data = df[selector]
    selectivity = data["selectivity"].reset_index(drop=True)[0]

    nrow = 2
    ncol = (len(DATASETS) + nrow - 1) // nrow
    fig, axs = plt.subplots(nrow, ncol, layout='constrained')
    for i, dataset in enumerate(DATASETS):
      for m in selected_builds:
        for b in [TEMPLATES[m].build.format(*_) for _ in selected_builds[m]]:
          marker = COMPASS_BUILD_MARKER_MAPPING.get(b, TWOD_RUNS[m].marker) if markers is None else markers[m]
          data_by_m_b = data[(data["method"] == m) & (data["build"] == b) & (data["dataset"] == dataset)]
          if selected_searches and m in selected_searches:
            for s in selected_searches[m]:
              data_by_m_b_s = data_by_m_b[data_by_m_b["run"].str.contains(s)]
              if data_by_m_b_s.size == 0: continue
              qps_recall = data_by_m_b_s[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, True])
              qps_recall = qps_recall.to_numpy()
              axs[i // ncol, i % ncol].plot(qps_recall[:, 0], qps_recall[:, 1])
              axs[i // ncol, i % ncol].scatter(qps_recall[:, 0], qps_recall[:, 1], label=f"{m}-{b}-{s}", marker=marker)
          else:
            qps_recall = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, True])
            qps_recall = qps_recall.to_numpy()
            axs[i // ncol, i % ncol].plot(qps_recall[:, 0], qps_recall[:, 1])
            axs[i // ncol, i % ncol].scatter(qps_recall[:, 0], qps_recall[:, 1], label=f"{m}-{b}", marker=marker)

          axs[i // ncol, i % ncol].set_xlabel('Recall')
          axs[i // ncol, i % ncol].set_ylabel('QPS')
          axs[i // ncol, i % ncol].set_title("{}, Selectivity-{}".format(dataset.capitalize(), selectivity))

    fig.set_size_inches(15, 10)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside right upper')
    fig.savefig(f"cherrypick2d_{K}/" + f"{subdir}" + f"All-{selectivity}-QPS-Recall.jpg", dpi=200)
    plt.close()


def draw_2d_comp_fixed_recall_by_selectivity(workloads, ranges, compare_by):
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
    "comp": float,
  }
  df = pd.read_csv(f"stats2d_{K}.csv", dtype=types)
  cutoff_recalls = [0.8, 0.9, 0.95]

  selectivities = [f"{int(w.split('-')[0]) * int(w.split('-')[1]) / 10000:.2f}" for w in ranges]

  for recall in cutoff_recalls:
    nrow = 2
    ncol = (len(workloads) + nrow - 1) // nrow
    fig, axs = plt.subplots(nrow, ncol, layout='constrained')
    for i, dataset in enumerate(workloads):
      axs[i // ncol][i % ncol].set_xticks(np.arange(len(selectivities)))
      axs[i // ncol][i % ncol].set_xticklabels(selectivities)
      axs[i // ncol][i % ncol].set_title(dataset.upper())
      axs[i // ncol][i % ncol].set_ylabel('# Comp')
      axs[i // ncol][i % ncol].set_xlabel('Selectivity')
      data = df[df["dataset"] == dataset]
      for m in workloads[dataset]:
        for b in [TEMPLATES[m].build.format(*b) for b in workloads[dataset][m]]:
          marker = COMPASS_BUILD_MARKER_MAPPING.get(b, TWOD_RUNS[m].marker)
          data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
          if data_by_m_b.size == 0: continue
          rec_sel_qps_comp = data_by_m_b[["recall", "selectivity", "qps", "comp"]].sort_values(["selectivity", "recall"])

          grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(recall)].groupby("selectivity", as_index=False)["qps"].max()
          grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(recall)].groupby("selectivity", as_index=False)["comp"].min()
          grouped_comp = grouped_comp[grouped_comp["selectivity"].isin(ranges)]
          grouped_qps = grouped_qps[grouped_qps["selectivity"].isin(ranges)]
          pos_s = np.array([
            bisect.bisect(selectivities, f"{int(w.split('-')[0]) * int(w.split('-')[1]) / 10000:.2f}") for w in grouped_qps["selectivity"]
          ]) - 1
          axs[i // ncol][i % ncol].plot(pos_s, grouped_comp["comp"])
          axs[i // ncol][i % ncol].scatter(pos_s, grouped_comp["comp"], label=f"{m}-{b}-{recall}", marker=marker)

    fig.set_size_inches(21, 12)
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper")
    fig.savefig(f"cherrypick2d_{K}/Recall-{recall:.2g}-{compare_by}-All-Comp.jpg", dpi=200)
    plt.close()


plt.rcParams.update({
  'font.size': 15,
  'legend.fontsize': 15,
  'axes.labelsize': 15,
  'axes.titlesize': 15,
  'figure.figsize': (10, 5),
})

evenhand_ranges = [f"{pcnt}-{pcnt}" for pcnt in [1, 5, 10, 30, 50, 80, 90]]
tot_mom_ranges = [f"{pcnt}-{pcnt}" for pcnt in [10, 30, 50, 80, 90]]


# Compare with baseline methods to reach recall
def compare_with_baseline_to_reach_recall():
  base_builds = {
    "CompassR": [
      CompassBuild(16, 200, 10000),  # CompassBuild(32, 200, 10000),
    ],
    "CompassIvf": [
      # CompassIvfBuild(1000),
      CompassIvfBuild(5000),
      CompassIvfBuild(10000),
    ],
    "CompassGraph": [CompassGraphBuild(32, 200)],
  }
  workloads = OrderedDict()
  for d in DATASETS:
    workloads[d] = base_builds
  # workloads["crawl"] = {
  #   "CompassR": [CompassBuild(16, 200, 20000)],
  #   "CompassIvf": [CompassIvfBuild(20000)],
  #   "CompassGraph": [CompassGraphBuild(32, 200)],
  # }

  draw_2d_comp_fixed_recall_by_selectivity(workloads, tot_mom_ranges, "ToT")
  searches = {"CompassR": [f"nrel_{nrel}" for nrel in [500, 1000]]}
  draw_2d_comp_wrt_recall_by_selectivity(evenhand_ranges, base_builds, searches, "baseline/")


compare_with_baseline_to_reach_recall()


# Compare comp with baseline methods
def compare_comp_with_baseline():
  base_builds = {
    "CompassR": [
      CompassBuild(16, 200, 10000),  # CompassBuild(32, 200, 10000),
    ],
    "CompassIvf": [
      # CompassIvfBuild(1000),
      CompassIvfBuild(5000),
      CompassIvfBuild(10000),
    ],
    "CompassGraph": [CompassGraphBuild(32, 200)],
  }
  searches = {"CompassR": [f"nrel_{nrel}" for nrel in [500, 1000]]}
  draw_2d_comp_wrt_recall_by_selectivity(evenhand_ranges, base_builds, searches, "baseline/")


# Compare the QPS between original and cluster-graph version
def compare_qps_with_cluster_graph():
  time_trial_methods = {
    "CompassR": [
      CompassBuild(16, 200, 5000),
      CompassBuild(16, 200, 10000),
      CompassBuild(32, 200, 5000),
      CompassBuild(32, 200, 10000),
    ],
    "CompassRCg": [
      CompassBuild(16, 200, 5000),
      CompassBuild(16, 200, 10000),
      CompassBuild(32, 200, 5000),
      CompassBuild(32, 200, 10000),
    ],
    "iRangeGraph2d": [iRangeGraphBuild(32, 200)]
  }
  time_trial_searches = {
    "CompassR": [f"nrel_{nrel}" for nrel in [500, 600, 800, 1000]],
    "CompassRCg": [f"nrel_{nrel}" for nrel in [500, 600, 800, 1000]],
  }
  time_trial_markers = {"CompassR": "o", "CompassRCg": "^", "iRangeGraph2d": "2"}
  draw_2d_qps_wrt_recall_by_selectivity(evenhand_ranges, time_trial_methods, time_trial_searches, time_trial_markers, "timing/")


# Compare with SotA methods to reach recall
def compare_with_sota_to_reach_recall():
  builds = {
    "iRangeGraph2d": [iRangeGraphBuild(32, 200)],
    "CompassR": [CompassBuild(16, 200, 10000)],
  }
  workloads = OrderedDict()
  for d in DATASETS:
    workloads[d] = builds
  # workloads["crawl"] = {
  #   "iRangeGraph2d": [iRangeGraphBuild(32, 200)],
  #   "CompassR": [CompassBuild(16, 200, 20000)],
  # }

  draw_2d_comp_fixed_recall_by_selectivity(workloads, tot_mom_ranges, "MoM")


compare_with_sota_to_reach_recall()


# Compare #Comp-Recall when using different efs
def compare_comp_varying_efs():
  methods = {
    "iRangeGraph2d": [iRangeGraphBuild(32, 200)],  # "Serf2d": [SerfBuild(32, 200, 500)],
    "CompassR": [CompassBuild(32, 200, 10000)],
  }
  searches = {"CompassR": [f"nrel_{nrel}" for nrel in [500, 1000]]}
  draw_2d_comp_wrt_recall_by_selectivity(TWOD_RANGES, methods, searches, "varying-efs/")


# Compare #Comp-Recall when using different nrel
def compare_comp_varying_nrel():
  methods = {
    "iRangeGraph2d": [iRangeGraphBuild(32, 200)],  # "Serf2d": [SerfBuild(32, 200, 500)],
    "CompassR": [CompassBuild(32, 200, 10000)],
  }
  searches = {"CompassR": [f"nrel_{nrel}" for nrel in [500, 600, 800, 1000]]}
  draw_2d_comp_wrt_recall_by_selectivity(TWOD_RANGES, methods, searches, "varying-nrel/")


# Compare #Comp-Recall when using different M
def compare_comp_varying_M():
  methods = {
    "iRangeGraph2d": [iRangeGraphBuild(32, 200)],  # "Serf2d": [SerfBuild(32, 200, 500)],
    "CompassR": [
      CompassBuild(16, 200, 10000),
      CompassBuild(32, 200, 10000),
    ],
  }
  searches = {"CompassR": [f"nrel_{nrel}" for nrel in [1000]]}
  draw_2d_comp_wrt_recall_by_selectivity(TWOD_RANGES, methods, searches, "varying-M/")


# Compare #Comp-Recall when using different nlist
def compare_comp_varying_nlist():
  methods = {
    "iRangeGraph2d": [iRangeGraphBuild(32, 200)],  # "Serf2d": [SerfBuild(32, 200, 500)],
    "CompassR": [
      CompassBuild(16, 200, 1000),
      CompassBuild(16, 200, 2000),
      CompassBuild(16, 200, 5000),
      CompassBuild(16, 200, 10000),
    ],
  }
  searches = {"CompassR": [f"nrel_{nrel}" for nrel in [100]]}
  draw_2d_comp_wrt_recall_by_selectivity(TWOD_RANGES, methods, searches, "varying-nlist/")


# Compare #Comp-Recall with SotA methods
def compare_comp_with_sota():
  methods = {
    "iRangeGraph2d": [iRangeGraphBuild(32, 200)],  # "Serf2d": [SerfBuild(32, 200, 500)],
    "CompassR": [
      CompassBuild(16, 200, 1000),
      CompassBuild(16, 200, 10000),
      CompassBuild(32, 200, 1000),
      CompassBuild(32, 200, 10000),
    ],
    "CompassRCg": [
      CompassBuild(16, 200, 1000),
      CompassBuild(16, 200, 10000),
      CompassBuild(32, 200, 1000),
      CompassBuild(32, 200, 10000),
    ],
  }
  searches = {
    "CompassR": [f"nrel_{nrel}" for nrel in [500, 1000]],
    "CompassRCg": [f"nrel_{nrel}" for nrel in [500, 1000]],
  }
  draw_2d_comp_wrt_recall_by_selectivity(evenhand_ranges, methods, searches)
