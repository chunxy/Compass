from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import json
import pandas as pd
import bisect

from config import *

K = 10
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
    if len(splits) == 5:  # CompassR
      l_bounds, r_bounds = splits[2].strip("{}").split(", "), splits[3].strip("{}").split(", ")
      l1, l2 = list(map(int, l_bounds))
      r1, r2 = list(map(int, r_bounds))
      # selectivity = (r1 - l1) * (r2 - l2) / int(splits[1]) / int(splits[1])
      selectivity = f"{(r1 - l1) / int(splits[1]) * 100:.0f}-{(r2 - l2)  / int(splits[1]) * 100:.0f}"
    else:  # iRangeGraph and Serf
      # selectivity = int(splits[1]) / int(splits[2])
      selectivity = f"{splits[1]}-{splits[2]}"
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

  selectors = [((df["dataset"] == d) & (df["selectivity"] == r)) for d in DATASETS for r in TWOD_RANGES]
  selected_methods = ["CompassIvf", "CompassRRBikmeans", "CompassRRCgBikmeans", "CompassGraph", "iRangeGraph2d"]

  for selector in selectors:
    if not selector.any(): continue
    data = df[selector]
    dataset = data["dataset"].reset_index(drop=True)[0]
    selectivity = data["selectivity"].reset_index(drop=True)[0]

    fig, axs = plt.subplots(2, 1, layout='constrained')
    for m in selected_methods:
      for b in data[data["method"] == m].build.unique():
        data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
        marker = COMPASS_BUILD_MARKER_MAPPING.get(b, TWOD_RUNS[m].marker)
        if m == "CompassRRBikmeans" or m == "CompassRRCgBikmeans" or m == "CompassGraph":
          for nrel in [100, 200]:
            data_by_m_b_nrel = data_by_m_b[data_by_m_b["run"].str.contains(f"nrel_{nrel}")]
            if data_by_m_b_nrel.size == 0: continue
            recall_qps = data_by_m_b_nrel[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
            recall_qps = recall_qps.to_numpy()
            axs[0].plot(recall_qps[:, 0], recall_qps[:, 1])
            axs[0].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}-nrel_{nrel}", marker=marker)

            recall_comp = data_by_m_b_nrel[["recall", "comp"]].sort_values(["recall", "comp"], ascending=[True, True])
            recall_comp = recall_comp.to_numpy()
            axs[1].plot(recall_comp[:, 0], recall_comp[:, 1])
            axs[1].scatter(recall_comp[:, 0], recall_comp[:, 1], label=f"{m}-{b}-nrel_{nrel}", marker=marker)
        else:
          recall_qps = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
          recall_qps = recall_qps.to_numpy()
          axs[0].plot(recall_qps[:, 0], recall_qps[:, 1])
          axs[0].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", marker=TWOD_RUNS[m].marker)
          recall_comp = data_by_m_b[["recall", "comp"]].sort_values(["recall", "comp"], ascending=[True, True])
          recall_comp = recall_comp.to_numpy()
          axs[1].plot(recall_comp[:, 0], recall_comp[:, 1])
          axs[1].scatter(recall_comp[:, 0], recall_comp[:, 1], label=f"{m}-{b}", marker=TWOD_RUNS[m].marker)

        axs[0].set_xlabel('Recall')
        axs[0].set_ylabel('QPS')
        axs[0].set_title(f"{dataset.upper()}, Selectivity-{selectivity}")
        axs[1].set_xlabel('Recall')
        axs[1].set_ylabel('# Comp')
        axs[1].set_title(f"{dataset.upper()}, Selectivity-{selectivity}")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper")
    fig.savefig(f"figures2d_{K}/{dataset.upper()}/{dataset.upper()}-{selectivity}-QPS-Recall.jpg", dpi=200)
    plt.close()


def draw_2d_qps_comp_wrt_recall_by_selectivity():
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

  selectors = [df["selectivity"] == r for r in TWOD_RANGES]
  selected_methods = ["CompassIvf", "CompassRRBikmeans", "CompassRRCgBikmeans", "CompassGraph", "iRangeGraph2d"]

  for selector in selectors:
    if not selector.any(): continue
    data = df[selector]
    selectivity = data["selectivity"].reset_index(drop=True)[0]

    fig, axs = plt.subplots(2, len(DATASETS), layout='constrained')
    for i, dataset in enumerate(DATASETS):
      for m in selected_methods:
        for b in data[data["method"] == m].build.unique():
          data_by_m_b = data[(data["method"] == m) & (data["build"] == b) & (data["dataset"] == dataset)]
          marker = COMPASS_BUILD_MARKER_MAPPING.get(b, TWOD_RUNS[m].marker)
          if m == "CompassRRBikmeans" or m == "CompassRRCgBikmeans":
            for nrel in [100, 200]:
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["run"].str.contains(f"nrel_{nrel}")]
              if data_by_m_b_nrel.size == 0: continue
              recall_qps = data_by_m_b_nrel[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
              recall_qps = recall_qps.to_numpy()
              axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1])
              axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}-{nrel}", marker=marker)

              recall_comp = data_by_m_b_nrel[["recall", "comp"]].sort_values(["recall", "comp"], ascending=[True, True])
              recall_comp = recall_comp.to_numpy()
              axs[1][i].plot(recall_comp[:, 0], recall_comp[:, 1])
              axs[1][i].scatter(recall_comp[:, 0], recall_comp[:, 1], label=f"{m}-{b}-{nrel}", marker=marker)
          else:
            recall_qps = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
            recall_qps = recall_qps.to_numpy()
            axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1])
            axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", marker=marker)

            recall_comp = data_by_m_b[["recall", "comp"]].sort_values(["recall", "comp"], ascending=[True, True])
            recall_comp = recall_comp.to_numpy()
            axs[1][i].plot(recall_comp[:, 0], recall_comp[:, 1])
            axs[1][i].scatter(recall_comp[:, 0], recall_comp[:, 1], label=f"{m}-{b}", marker=marker)

          axs[0][i].set_xlabel('Recall')
          axs[0][i].set_ylabel('QPS')
          axs[0][i].set_title("{}, Selectivity-{}".format(dataset.capitalize(), selectivity))
          axs[1][i].set_xlabel('Recall')
          axs[1][i].set_ylabel('# Comp')
          axs[1][i].set_title("{}, Selectivity-{}".format(dataset.capitalize(), selectivity))

    fig.set_size_inches(35, 20)
    # handles, labels = axs[0][0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='outside right upper')
    unique_labels = {}
    for ax in axs.flat:
      handles, labels = ax.get_legend_handles_labels()
      for handle, label in zip(handles, labels):
        if label not in unique_labels:
          unique_labels[label] = handle
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='outside right upper')
    fig.savefig(f"figures2d_{K}/All-{selectivity}-QPS-Comp-Recall.jpg", dpi=200)
    plt.close()


def draw_2d_qps_comp_fixed_recall_by_selectivity(selected_methods, compare_by):
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

  pcnts = [10, 30, 50, 80]
  selectivities = [f"{pcnt}-{pcnt}" for pcnt in pcnts]

  for recall in cutoff_recalls:
    fig, axs = plt.subplots(2, len(DATASETS), layout='constrained', sharex=True)
    for i, dataset in enumerate(DATASETS):
      axs[0][i].set_xticks(np.arange(len(selectivities)))
      ticklabels = [f"{pcnt * pcnt / 10000:.2f}" for pcnt in pcnts]
      axs[0][i].set_xticklabels(ticklabels)
      axs[0][i].set_title(dataset.upper())
      axs[0][i].set_ylabel('QPS')
      axs[1][i].set_ylabel('# Comp')
      axs[1][i].set_xlabel('Selectivity')
      data = df[df["dataset"] == dataset]
      for m in selected_methods:
        for b in data[data["method"] == m].build.unique():
          data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
          marker = COMPASS_BUILD_MARKER_MAPPING.get(b, TWOD_RUNS[m].marker)
          if m == "CompassRRBikmeans" or m == "CompassRRCgBikmeans":
            for nrel in [100, 200]:
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["run"].str.contains(f"nrel_{nrel}")]
              if data_by_m_b_nrel.size == 0: continue
              rec_sel_qps_comp = data_by_m_b_nrel[["recall", "selectivity", "qps", "comp"]].sort_values(["selectivity", "recall"])
              grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(recall - 0.05)].groupby("selectivity", as_index=False)["qps"].max()
              grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(recall - 0.05)].groupby("selectivity", as_index=False)["comp"].min()
              pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
              axs[0][i].plot(pos_s, grouped_qps["qps"])
              axs[0][i].scatter(pos_s, grouped_qps["qps"], label=f"{m}-{b}-{recall}-{nrel}", marker=marker)
              axs[1][i].plot(pos_s, grouped_comp["comp"])
              axs[1][i].scatter(pos_s, grouped_comp["comp"], label=f"{m}-{b}-{recall}-{nrel}", marker=marker)
          else:
            rec_sel_qps_comp = data_by_m_b[["recall", "selectivity", "qps", "comp"]].sort_values(["selectivity", "recall"])
            grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(recall - 0.05)].groupby("selectivity", as_index=False)["qps"].max()
            grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(recall - 0.05)].groupby("selectivity", as_index=False)["comp"].min()
            pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
            axs[0][i].plot(pos_s, grouped_qps["qps"])
            axs[0][i].scatter(pos_s, grouped_qps["qps"], label=f"{m}-{b}-{recall}", marker=marker)
            axs[1][i].plot(pos_s, grouped_comp["comp"])
            axs[1][i].scatter(pos_s, grouped_comp["comp"], label=f"{m}-{b}-{recall}", marker=marker)

    fig.set_size_inches(45, 20)
    # handles, labels = axs[0][0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="outside right upper")
    unique_labels = {}
    for ax in axs.flat:
      handles, labels = ax.get_legend_handles_labels()
      for handle, label in zip(handles, labels):
        if label not in unique_labels:
          unique_labels[label] = handle
    fig.legend(unique_labels.values(), unique_labels.keys(), loc='outside right upper')
    fig.savefig(f"figures2d_{K}/Recall-{recall:.2g}-{compare_by}-All-QPS-Comp.jpg", dpi=200)
    plt.close()


plt.rcParams.update({
  'font.size': 15,
  'legend.fontsize': 12,
  'axes.labelsize': 15,
  'axes.titlesize': 15,
  'figure.figsize': (10, 15),
})
summarize_2d()

draw_2d_qps_comp_wrt_recall_by_dataset_selectivity()
draw_2d_qps_comp_wrt_recall_by_selectivity()

selected_methods = ["CompassRRBikmeans", "CompassRRCgBikmeans", "CompassIvf", "CompassGraph", "iRangeGraph2d"]
draw_2d_qps_comp_fixed_recall_by_selectivity(selected_methods, "ToT")

