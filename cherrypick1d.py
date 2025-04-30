from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import bisect
from collections import OrderedDict

from config import *

K = 10
LOGS = Path(LOGS_TMPL.format(K))


def draw_1d_comp_wrt_recall_by_selectivity(selected_builds, selected_searches=None, purpose=""):
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
  df = pd.read_csv(f"stats1d_{K}.csv", dtype=types)

  selectors = [df["selectivity"] == r for r in ONED_PASSRATES]

  for selector in selectors:
    if not selector.any(): continue
    data = df[selector]
    selectivity = float(data["selectivity"].reset_index(drop=True)[0])

    nrow = 2
    ncol = (len(DATASETS) + nrow - 1) // nrow
    fig, axs = plt.subplots(nrow, ncol, layout='constrained')
    for i, dataset in enumerate(DATASETS):
      for m in selected_builds:
        for b in [TEMPLATES[m].build.format(*_) for _ in selected_builds[m]]:
          marker = COMPASS_BUILD_MARKER_MAPPING.get(b, ONED_RUNS[m].marker)
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
          axs[i // ncol, i % ncol].set_title("{}, Selectivity-{:.1%}".format(dataset.capitalize(), selectivity))

    fig.set_size_inches(15, 10)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside right upper')
    fig.savefig(f"cherrypick_{K}/" + f"{purpose}" + f"All-{selectivity:.1%}-Comp-Recall.jpg", dpi=200)
    plt.close()


def draw_1d_qps_wrt_recall_by_selectivity(selected_builds, selected_searches=None, markers=None, purpose=""):
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
  df = pd.read_csv(f"stats1d_{K}.csv", dtype=types)

  selectors = [df["selectivity"] == r for r in ONED_PASSRATES]

  for selector in selectors:
    if not selector.any(): continue
    data = df[selector]
    selectivity = float(data["selectivity"].reset_index(drop=True)[0])

    nrow = 2
    ncol = (len(DATASETS) + nrow - 1) // nrow
    fig, axs = plt.subplots(nrow, ncol, layout='constrained')
    for i, dataset in enumerate(DATASETS):
      for m in selected_builds:
        for b in [TEMPLATES[m].build.format(*_) for _ in selected_builds[m]]:
          marker = COMPASS_BUILD_MARKER_MAPPING.get(b, ONED_RUNS[m].marker) if markers is None else markers[m]
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
          axs[i // ncol, i % ncol].set_title("{}, Selectivity-{:.1%}".format(dataset.capitalize(), selectivity))

    fig.set_size_inches(15, 10)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside right upper')
    fig.savefig(f"cherrypick_{K}/" + f"{purpose}" + f"All-{selectivity:.1%}-QPS-Recall.jpg", dpi=200)
    plt.close()


def draw_1d_comp_fixed_recall_by_selectivity(selected_datasets, compare_by):
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
  df = pd.read_csv(f"stats1d_{K}.csv", dtype=types)
  cutoff_recalls = [0.8, 0.9, 0.95]

  selectivities = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.3", "0.4", "0.6", "0.8", "1.0"]

  for recall in cutoff_recalls:
    nrow = 2 if len(selected_datasets) >= 4 else 1
    ncol = (len(selected_datasets) + nrow - 1) // nrow
    fig, axs = plt.subplots(nrow, ncol, layout='constrained')
    axs = axs.flat
    for i, dataset in enumerate(selected_datasets):
      axs[i].set_xticks(np.arange(len(selectivities)))
      axs[i].set_xticklabels(selectivities)
      axs[i].set_title(dataset.upper())
      axs[i].set_ylabel('# Comp')
      axs[i].set_xlabel('Selectivity')
      data = df[df["dataset"] == dataset]
      for m in selected_datasets[dataset]:
        for b in [TEMPLATES[m].build.format(*b) for b in selected_datasets[dataset][m]["build"]]:
          marker = COMPASS_BUILD_MARKER_MAPPING.get(b, ONED_RUNS[m].marker)
          data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
          if data_by_m_b.size == 0: continue
          if "search" in selected_datasets[dataset][m]:
            for s in selected_datasets[dataset][m]["search"]:
              data_by_m_b_s = data_by_m_b[data_by_m_b["run"].str.contains(s)]
              if data_by_m_b_s.size == 0: continue
              rec_sel_qps_comp = data_by_m_b_s[["recall", "selectivity", "qps", "comp"]].sort_values(["selectivity", "recall"])

              grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(recall)].groupby("selectivity", as_index=False)["qps"].max()
              grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(recall)].groupby("selectivity", as_index=False)["comp"].min()
              grouped_qps = grouped_qps[grouped_qps["selectivity"].isin(selectivities)]
              grouped_comp = grouped_comp[grouped_comp["selectivity"].isin(selectivities)]
              pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
              axs[i].plot(pos_s, grouped_comp["comp"])
              axs[i].scatter(pos_s, grouped_comp["comp"], label=f"{m}-{b}-{recall}-{s}", marker=marker)
          else:
            rec_sel_qps_comp = data_by_m_b[["recall", "selectivity", "qps", "comp"]].sort_values(["selectivity", "recall"])

            grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(recall)].groupby("selectivity", as_index=False)["qps"].max()
            grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(recall)].groupby("selectivity", as_index=False)["comp"].min()
            grouped_qps = grouped_qps[grouped_qps["selectivity"].isin(selectivities)]
            grouped_comp = grouped_comp[grouped_comp["selectivity"].isin(selectivities)]
            pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
            axs[i].plot(pos_s, grouped_comp["comp"])
            axs[i].scatter(pos_s, grouped_comp["comp"], label=f"{m}-{b}-{recall}", marker=marker)

    fig.set_size_inches(21, 12)
    # handles, labels = axs[0][0].get_legend_handles_labels()
    unique_labels = {}
    for ax in axs:
      handles, labels = ax.get_legend_handles_labels()
      for handle, label in zip(handles, labels):
        if label not in unique_labels:
          unique_labels[label] = handle
    fig.legend(unique_labels.values(), unique_labels.keys(), loc="outside right upper")
    # fig.legend(handles, labels, loc="outside right upper")
    fig.savefig(f"cherrypick_{K}/Recall-{recall:.2g}-{compare_by}-All-Comp.jpg", dpi=200)
    plt.close()


def draw_1d_by_selected_dataset_adverse():
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
  df = pd.read_csv(f"stats1d_{K}.csv", dtype=types)

  SELECTED_DATASETS = ["sift", "crawl", "audio", "video"]
  SELECTED_PASSRATES = ["0.01", "0.5", "1.0"]
  SELECTED_BUILDS = {
    "CompassR1d": CompassBuild(32, 200, 1000),
    "iRangeGraph": iRangeGraphBuild(32, 200),
    "Serf": SerfBuild(32, 200, 500),
  }
  for passrate in SELECTED_PASSRATES:
    fig, axs = plt.subplots(2, len(SELECTED_DATASETS), layout='constrained')
    fig.set_size_inches(20, 10)
    for i, d in enumerate(SELECTED_DATASETS):
      data = df[df["selectivity"] == passrate]
      data = data[data["dataset"] == d]
      data = data[data["recall"] >= 0.8]
      selectivity = float(passrate)
      for m in SELECTED_BUILDS:
        b = TEMPLATES[m].build.format(*SELECTED_BUILDS[m])
        recall_qps = data[(data["method"] == m) & (data["build"] == b)][["recall", "qps"]].sort_values(["recall", "qps"],
                                                                                                        ascending=[True, False]).to_numpy()
        axs[0, i].plot(recall_qps[:, 0], recall_qps[:, 1])
        axs[0, i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", marker=ONED_RUNS[m].marker)
        axs[0, i].set_xlabel('Recall')
        axs[0, i].set_ylabel('QPS')
        axs[0, i].set_title("{}, Passrate-{:.1%}".format(d.capitalize(), selectivity))

        recall_ncomp = data[(data["method"] == m) & (data["build"] == b)][["recall", "comp"]].sort_values(["recall", "comp"],
                                                                                                          ascending=[True, True]).to_numpy()
        axs[1, i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1])
        axs[1, i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1], label=f"{m}-{b}", marker=ONED_RUNS[m].marker)
        axs[1, i].set_xlabel('Recall')
        axs[1, i].set_ylabel('#Comp')
        axs[1, i].set_title("{}, Passrate-{:.1%}".format(d.capitalize(), selectivity))
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center")
    fig.savefig(f"cherrypick_{K}/Adverse-Selected-Dataset-{selectivity:.1%}.jpg", dpi=200)
    fig.clf()


def draw_1d_by_selected_dataset_favorable():
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
  df = pd.read_csv(f"stats1d_{K}.csv", dtype=types)

  SELECTED_DATASETS = ["sift", "crawl", "audio", "video"]
  SELECTED_PASSRATES = ["0.01", "0.5", "1.0"]
  SELECTED_BUILDS = {
    "CompassR1d": CompassBuild(32, 200, 1000),
    "iRangeGraph": iRangeGraphBuild(16, 200),
    "Serf": SerfBuild(16, 200, 500),
  }
  for passrate in SELECTED_PASSRATES:
    fig, axs = plt.subplots(2, len(SELECTED_DATASETS), layout='constrained')
    fig.set_size_inches(20, 10)
    for i, d in enumerate(SELECTED_DATASETS):
      data = df[df["selectivity"] == passrate]
      data = data[data["dataset"] == d]
      data = data[data["recall"] >= 0.8]
      selectivity = float(passrate)
      for m in SELECTED_BUILDS:
        b = TEMPLATES[m].build.format(*SELECTED_BUILDS[m])
        recall_qps = data[(data["method"] == m) & (data["build"] == b)][["recall", "qps"]].sort_values(["recall", "qps"],
                                                                                                        ascending=[True, False]).to_numpy()
        axs[0, i].plot(recall_qps[:, 0], recall_qps[:, 1])
        axs[0, i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", marker=ONED_RUNS[m].marker)
        axs[0, i].set_xlabel('Recall')
        axs[0, i].set_ylabel('QPS')
        axs[0, i].set_title("{}, Passrate-{:.1%}".format(d.capitalize(), selectivity))

        recall_ncomp = data[(data["method"] == m) & (data["build"] == b)][["recall", "comp"]].sort_values(["recall", "comp"],
                                                                                                          ascending=[True, True]).to_numpy()
        axs[1, i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1])
        axs[1, i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1], label=f"{m}-{b}", marker=ONED_RUNS[m].marker)
        axs[1, i].set_xlabel('Recall')
        axs[1, i].set_ylabel('#Comp')
        axs[1, i].set_title("{}, Passrate-{:.1%}".format(d.capitalize(), selectivity))
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center")
    fig.savefig(f"cherrypick_{K}/Favorable-Selected-Dataset-{selectivity:.1%}.jpg", dpi=200)
    fig.clf()


plt.rcParams.update({
  'font.size': 15,
  'legend.fontsize': 15,
  'axes.labelsize': 15,
  'axes.titlesize': 15,
  'figure.figsize': (10, 5),
})


# Compare with baseline methods to reach recall
def compare_comp_with_baseline():
  build_search = {
    "CompassR1d": {
      "build": [CompassBuild(16, 200, 5000), CompassBuild(16, 200, 10000)],
      "search": [f"nrel_{nrel}" for nrel in [500, 800]],
    },
    "CompassIvf1d": {
      "build": [CompassBuild(16, 200, 10000)],
    },
    "CompassGraph1d": {
      "build": [CompassBuild(16, 200, 10000)],
    },
  }
  datasets = OrderedDict()
  for d in DATASETS:
    datasets[d] = build_search
  draw_1d_comp_fixed_recall_by_selectivity(datasets, "ToT")
  base_builds = {
    "CompassR1d": [
      CompassBuild(16, 200, 5000),
      CompassBuild(16, 200, 10000),
    ],
    "CompassIvf1d": [CompassIvfBuild(1000), CompassIvfBuild(5000), CompassIvfBuild(10000)],
    "CompassGraph1d": [CompassGraphBuild(32, 200)],
  }
  selected_searches = {"CompassR1d": [f"nrel_{nrel}" for nrel in [500, 800]]}
  draw_1d_comp_wrt_recall_by_selectivity(base_builds, selected_searches, "baseline/")


# Compare the QPS between original and cluster-graph version
def compare_qps_with_cluster_graph():
  build_search = {
    "CompassR1d": {
      "build": [CompassBuild(16, 200, 5000), CompassBuild(16, 200, 10000)],
      "search": [f"nrel_{nrel}" for nrel in [500]],
    },
    "CompassRR1d": {
      "build": [CompassBuild(16, 200, 10000)],
      "search": [f"nrel_{nrel}" for nrel in [500]],
    },
    "CompassRRCg1d": {
      "build": [CompassBuild(16, 200, 10000)],
      "search": [f"nrel_{nrel}" for nrel in [500]],
    },
    "iRangeGraph": {
      "build": [iRangeGraphBuild(32, 200)]
    }
  }
  time_trial_methods = {
    "CompassR1d": [
      # CompassBuild(16, 200, 1000),
      # CompassBuild(16, 200, 5000),
      CompassBuild(16, 200, 10000),
    ],
    "CompassRR1d": [
      # CompassBuild(16, 200, 1000),
      # CompassBuild(16, 200, 5000),
      CompassBuild(16, 200, 10000),
    ],
    "CompassRRCg1d": [
      # CompassBuild(16, 200, 1000),
      # CompassBuild(16, 200, 5000),
      CompassBuild(16, 200, 10000),
    ],
    "iRangeGraph": [iRangeGraphBuild(32, 200)]
  }
  time_trial_searches = {
    "CompassR1d": [f"nrel_{nrel}" for nrel in [500]],
    "CompassRR1d": [f"nrel_{nrel}" for nrel in [500]],
    "CompassRRCg1d": [f"nrel_{nrel}" for nrel in [500]],
  }
  time_trial_markers = {
    "CompassR1d": "o",
    "CompassRR1d": "o",
    "CompassRRCg1d": "^",
    "iRangeGraph": "2",
  }
  draw_1d_qps_wrt_recall_by_selectivity(time_trial_methods, time_trial_searches, time_trial_markers, "timing/")


compare_qps_with_cluster_graph()


# Compare with SotA methods to reach recall
def compare_comp_with_sota_to_reach_recall():
  build_search = {
    "iRangeGraph": {
      "build": [iRangeGraphBuild(32, 200)]
    },
    "Serf": {
      "build": [SerfBuild(32, 200, 500)]
    },
    "CompassR1d": {
      "build": [CompassBuild(16, 200, 10000)],
      "search": [f"nrel_{nrel}" for nrel in [500]],
    },
    # "CompassRR1dBikmeans": {
    #   "build": [CompassBuild(16, 200, 10000)],
    #   "search": [f"nrel_{nrel}" for nrel in [500]],
    # },
  }
  selected_datasets = OrderedDict()
  selected_datasets["sift"] = build_search
  selected_datasets["audio"] = build_search
  selected_datasets["crawl"] = {
    "iRangeGraph": {
      "build": [iRangeGraphBuild(32, 200)]
    },
    "Serf": {
      "build": [SerfBuild(32, 200, 500)]
    },
    "CompassR1d": {
      "build": [CompassBuild(16, 200, 20000)],
      "search": [f"nrel_{nrel}" for nrel in [1000]],
    },
    # "CompassRR1dBikmeans": {
    #   "build": [CompassBuild(16, 200, 20000)],
    #   "search": [f"nrel_{nrel}" for nrel in [1000]],
    # },
  }
  draw_1d_comp_fixed_recall_by_selectivity(selected_datasets, "MoM")


compare_comp_with_sota_to_reach_recall()


# Compare different clustering algorithms
def compare_comp_varying_clustering_algo():
  builds = {
    "CompassR1d": [CompassBuild(16, 200, 10000)],
    "CompassRR1d": [CompassBuild(16, 200, 10000)],
    "CompassRR1dBikmeans": [CompassBuild(16, 200, 10000)],
    "CompassRR1dKmedoids": [CompassBuild(16, 200, 10000)]
  }
  searches = {
    "CompassR1d": [f"nrel_{nrel}" for nrel in [500, 800]],
    "CompassRR1d": [f"nrel_{nrel}" for nrel in [500, 800]],
    "CompassRR1dBikmeans": [f"nrel_{nrel}" for nrel in [500, 800]],
    "CompassRR1dKmedoids": [f"nrel_{nrel}" for nrel in [500, 800]],
  }
  draw_1d_comp_wrt_recall_by_selectivity(builds, searches, "clustering/")

  datasets = OrderedDict()
  for d in DATASETS:
    datasets[d] = {}
    for m in builds:
      datasets[d]["build"] = builds[m]
      datasets[d]["search"] = searches[m]
  crawl_builds = {
    "CompassR1d": [CompassBuild(16, 200, 20000)],
    "CompassRR1d": [CompassBuild(16, 200, 20000)],
    "CompassRR1dBikmeans": [CompassBuild(16, 200, 20000)],
    "CompassRR1dKmedoids": [CompassBuild(16, 200, 20000)]
  }
  crawl_searches = {
    "CompassR1d": [f"nrel_{nrel}" for nrel in [1000]],
    "CompassRR1d": [f"nrel_{nrel}" for nrel in [1000]],
    "CompassRR1dBikmeans": [f"nrel_{nrel}" for nrel in [1000]],
    "CompassRR1dKmedoids": [f"nrel_{nrel}" for nrel in [1000]],
  }
  for m in crawl_builds:
    datasets["crawl"]["build"] = crawl_builds[m]
    datasets["crawl"]["search"] = crawl_searches[m]

  draw_1d_comp_fixed_recall_by_selectivity(datasets, "CoC")


# Compare #Comp-Recall when using different efs
def compare_comp_varying_efs():
  methods = {
    "iRangeGraph": [iRangeGraphBuild(32, 200)],
    "Serf": [SerfBuild(32, 200, 500)],
    "CompassR1d": [CompassBuild(32, 200, 10000), CompassBuild(32, 200, 1000)],
  }
  searches = {"CompassR1d": [f"nrel_{nrel}" for nrel in [500, 800, 1000]]}
  draw_1d_comp_wrt_recall_by_selectivity(methods, searches, "varying-efs/")


# Compare #Comp-Recall when using different nrel
def compare_comp_varying_nrel():
  methods = {
    "iRangeGraph": [iRangeGraphBuild(32, 200)],
    "Serf": [SerfBuild(32, 200, 500)],
    "CompassR1d": [CompassBuild(32, 200, 1000)],
  }
  searches = {"CompassR1d": [f"nrel_{nrel}" for nrel in [500, 800, 1000]]}
  draw_1d_comp_wrt_recall_by_selectivity(methods, searches, "varying-nrel/")


# Compare #Comp-Recall when using different M
def compare_comp_varying_M():
  methods = {
    "iRangeGraph": [iRangeGraphBuild(32, 200)],
    "Serf": [SerfBuild(32, 200, 500)],
    "CompassR1d": [
      CompassBuild(16, 200, 10000),
      CompassBuild(32, 200, 10000),
    ],
    "CompassRCg1d": [
      CompassBuild(16, 200, 10000),
      CompassBuild(32, 200, 10000),
    ],
  }
  searches = {"CompassR1d": [f"nrel_{nrel}" for nrel in [500]]}
  draw_1d_comp_wrt_recall_by_selectivity(methods, searches, "varying-M/")


# Compare #Comp-Recall when using different nlist
def compare_comp_varying_nlist():
  methods = {
    "iRangeGraph": [iRangeGraphBuild(32, 200)],
    "Serf": [SerfBuild(32, 200, 500)],
    "CompassR1d": [
      CompassBuild(16, 200, 1000),
      CompassBuild(16, 200, 2000),
      CompassBuild(16, 200, 5000),
      CompassBuild(16, 200, 10000),
    ],
  }
  searches = {"CompassR1d": [f"nrel_{nrel}" for nrel in [500]]}
  draw_1d_comp_wrt_recall_by_selectivity(methods, searches, "varying-nlist/")


# Compare #Comp-Recall with SotA methods
def compare_comp_with_sota():
  methods = {
    "iRangeGraph": [iRangeGraphBuild(8, 200), iRangeGraphBuild(32, 200)],
    "Serf": [SerfBuild(32, 200, 500)],
    "CompassR1d": [
      CompassBuild(16, 200, 10000),
    ],
    "CompassRR1d": [
      CompassBuild(16, 200, 10000),
    ],
  }
  searches = {
    "CompassR1d": [f"nrel_{nrel}" for nrel in [500, 1000]],
    "CompassRR1d": [f"nrel_{nrel}" for nrel in [500, 1000]],
  }
  draw_1d_comp_wrt_recall_by_selectivity(methods, searches)


compare_comp_with_sota()
