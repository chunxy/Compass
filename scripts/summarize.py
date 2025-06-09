import bisect
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
  compass_args,
  da_interval,
  da_range,
  da_sel,
  DATASETS,
  m_marker,
  m_param,
  m_workload,
  METHODS,
)

LOG_ROOT = Path("/home/chunxy/repos/Compass/logs_10")

types = {
  "path": str,
  "method": str,
  "workload": str,
  "dataset": str,
  "selectivity": str,
  "range": str,
  "build": str,
  "search": str,
  "recall": float,
  "qps": float,
  "ncomp": float,
}


def summarize():
  for da, intervals in da_interval.items():
    entries = []
    for m in METHODS:
      for d in DATASETS:
        for itvl in intervals:
          w = m_workload[m].format(d, *map(lambda ele: "-".join(map(str, ele)), itvl))
          bt = "_".join([f"{bp}_{{}}" for bp in m_param[m]["build"]])
          st = "_".join([f"{sp}_{{}}" for sp in m_param[m]["search"]])
          for ba in product(*[compass_args[bp] for bp in m_param[m]["build"]]):
            b = bt.format(*ba)
            for sa in product(*[compass_args[sp] for sp in m_param[m]["search"]]):
              s = st.format(*sa)
              nrg = "-".join([f"{(r - l) // 100}" for l, r in zip(*itvl)])  # noqa: E741
              path = LOG_ROOT / m / w / b / s
              if path.exists():
                entries.append((path, m, w, d, nrg, b, s))
    df = pd.DataFrame.from_records(entries, columns=[
      "path",
      "method",
      "workload",
      "dataset",
      "range",
      "build",
      "search",
    ], index="path")

    sel, rec, qps, ncomp = [], [], [], []
    for e in entries:
      jsons = list(e[0].glob("*.json"))
      if len(jsons) == 0:
        df = df.drop(e[0])
        continue
      jsons.sort()
      with open(jsons[-1]) as f:
        stat = json.load(f)
        sel.append(f'{stat["aggregated"]["selectivity"]:.2f}')
        rec.append(stat["aggregated"]["recall"])
        qps.append(stat["aggregated"]["qps"])
        ncomp.append(stat["aggregated"]["num_computations"])
    df["selectivity"] = sel
    df["recall"] = rec
    df["qps"] = qps
    df["ncomp"] = ncomp

    df.to_csv(f"stats-{da}d.csv")


def draw_qps_comp_wrt_recall_by_dataset_selectivity(da, datasets, methods, *, d_m_b={}, m_b_s={}, prefix="figures"):
  df = pd.read_csv(f"stats-{da}d.csv", dtype=types)

  for d in datasets:
    for rg in da_range[da]:
      selector = ((df["dataset"] == d) & (df["range"] == rg))
      if not selector.any():
        continue

      data = df[selector]
      sel = float(data["selectivity"].unique()[0])
      fig, axs = plt.subplots(1, 2, layout='constrained')
      for m in methods:
        marker = m_marker[m]
        for b in d_m_b.get(d, {}).get(m, data[data["method"] == m].build.unique()):
          data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
          if m.startswith("Compass"):
            for nrel in compass_args["nrel"]:
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].str.contains(f"nrel_{nrel}")]
              recall_qps = data_by_m_b_nrel[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
              recall_qps = recall_qps.to_numpy()
              axs[0].plot(recall_qps[:, 0], recall_qps[:, 1])
              axs[0].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}-nrel_{nrel}", marker=marker)

              recall_comp = data_by_m_b_nrel[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
              recall_comp = recall_comp.to_numpy()
              axs[1].plot(recall_comp[:, 0], recall_comp[:, 1])
              axs[1].scatter(recall_comp[:, 0], recall_comp[:, 1], label=f"{m}-{b}-nrel_{nrel}", marker=marker)
          else:
            recall_qps = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
            recall_qps = recall_qps.to_numpy()
            axs[0].plot(recall_qps[:, 0], recall_qps[:, 1])
            axs[0].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", marker=marker)

            recall_comp = data_by_m_b[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
            recall_comp = recall_comp.to_numpy()
            axs[1].plot(recall_comp[:, 0], recall_comp[:, 1])
            axs[1].scatter(recall_comp[:, 0], recall_comp[:, 1], label=f"{m}-{b}", marker=marker)

          axs[0].set_xlabel('Recall')
          axs[0].set_ylabel('QPS')
          axs[0].set_title("{}, Selectivity-{:.1%}".format(d.capitalize(), sel))
          axs[1].set_xlabel('Recall')
          axs[1].set_ylabel('# Comp')
          axs[1].set_title("{}, Selectivity-{:.1%}".format(d.capitalize(), sel))

      fig.set_size_inches(12, 6)
      unique_labels = {}
      for ax in axs.flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
          if label not in unique_labels:
            unique_labels[label] = handle
      fig.legend(unique_labels.values(), unique_labels.keys(), loc="outside right upper")
      fig.savefig(f"{prefix}{da}d-10/{d.upper()}/{d.upper()}-{rg}-QPS-Comp-Recall.jpg", dpi=200)
      plt.close()


def draw_qps_comp_wrt_recall_by_selectivity(da, datasets, methods, *, d_m_b={}, m_b_s={}, prefix="figures"):
  df = pd.read_csv(f"stats-{da}d.csv", dtype=types)

  for rg in da_range[da]:
    fig, axs = plt.subplots(2, len(datasets), layout='constrained')
    for i, d in enumerate(datasets):
      selector = ((df["dataset"] == d) & (df["range"] == rg))
      if not selector.any():
        continue

      data = df[selector]
      sel = float(data["selectivity"].unique()[0])
      for m in methods:
        marker = m_marker[m]
        for b in d_m_b.get(d, {}).get(m, data[data["method"] == m].build.unique()):
          data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
          if m.startswith("Compass"):
            for nrel in compass_args["nrel"]:
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].str.contains(f"nrel_{nrel}")]
              recall_qps = data_by_m_b_nrel[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
              recall_qps = recall_qps.to_numpy()
              axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1])
              axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}-nrel_{nrel}", marker=marker)

              recall_comp = data_by_m_b_nrel[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
              recall_comp = recall_comp.to_numpy()
              axs[1][i].plot(recall_comp[:, 0], recall_comp[:, 1])
              axs[1][i].scatter(recall_comp[:, 0], recall_comp[:, 1], label=f"{m}-{b}-nrel_{nrel}", marker=marker)
          else:
            recall_qps = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
            recall_qps = recall_qps.to_numpy()
            axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1])
            axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", marker=marker)

            recall_comp = data_by_m_b[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
            recall_comp = recall_comp.to_numpy()
            axs[1][i].plot(recall_comp[:, 0], recall_comp[:, 1])
            axs[1][i].scatter(recall_comp[:, 0], recall_comp[:, 1], label=f"{m}-{b}", marker=marker)

          axs[0][i].set_xlabel('Recall')
          axs[0][i].set_ylabel('QPS')
          axs[0][i].set_title("{}, Selectivity-{:.1%}".format(d.capitalize(), sel))
          axs[1][i].set_xlabel('Recall')
          axs[1][i].set_ylabel('# Comp')
          axs[1][i].set_title("{}, Selectivity-{:.1%}".format(d.capitalize(), sel))

      fig.set_size_inches(20, 6)
      unique_labels = {}
      for ax in axs.flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
          if label not in unique_labels:
            unique_labels[label] = handle
      fig.legend(unique_labels.values(), unique_labels.keys(), loc="outside right upper")
      fig.savefig(f"{prefix}{da}d-10/All-{rg}-QPS-Comp-Recall.jpg", dpi=200)
      plt.close()


def draw_qps_comp_fixed_recall_by_dataset_selectivity(da, datasets, methods, anno, *, d_m_b={}, m_b_s={}, prefix="figures"):
  df = pd.read_csv(f"stats-{da}d.csv", dtype=types)
  recall_thresholds = [0.8, 0.9, 0.95]
  selectivities = da_sel[da]

  for d in datasets:
    for rec in recall_thresholds:
      fig, axs = plt.subplots(1, 2, layout='constrained')

      data = df[df["dataset"] == d]
      for m in methods:
        marker = m_marker[m]
        for b in d_m_b.get(d, {}).get(m, data[data["method"] == m].build.unique()):
          data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
          if m.startswith("Compass"):
            for nrel in compass_args["nrel"]:
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].str.contains(f"nrel_{nrel}")]
              rec_sel_qps_comp = data_by_m_b_nrel[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
              grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
              grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
              pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
              axs[0].plot(pos_s, grouped_qps["qps"])
              axs[0].scatter(pos_s, grouped_qps["qps"], label=f"{m}-{b}-{rec}-{nrel}", marker=marker)
              axs[1].plot(pos_s, grouped_comp["ncomp"])
              axs[1].scatter(pos_s, grouped_comp["ncomp"], label=f"{m}-{b}-{rec}-{nrel}", marker=marker)
          else:
            rec_sel_qps_comp = data_by_m_b[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
            grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
            grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
            pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
            axs[0].plot(pos_s, grouped_qps["qps"])
            axs[0].scatter(pos_s, grouped_qps["qps"], label=f"{m}-{b}-{rec}", marker=marker)
            axs[1].plot(pos_s, grouped_comp["ncomp"])
            axs[1].scatter(pos_s, grouped_comp["ncomp"], label=f"{m}-{b}-{rec}", marker=marker)

      axs[0].set_xticks(np.arange(len(selectivities)))
      axs[0].set_xticklabels(selectivities)
      axs[0].set_xlabel('Selectivity')
      axs[0].set_ylabel('QPS')
      axs[0].set_title(f"{d.upper()}, Recall-{rec:.3g}")
      axs[1].set_xticks(np.arange(len(selectivities)))
      axs[1].set_xticklabels(selectivities)
      axs[1].set_ylabel('# Comp')
      axs[1].set_xlabel('Selectivity')
      axs[1].set_title(f"{d.upper()}, Recall-{rec:.3g}")

      fig.set_size_inches(14, 5)
      handles, labels = axs[0].get_legend_handles_labels()
      fig.legend(handles, labels, loc="outside right upper")
      fig.savefig(f"{prefix}{da}d-10/{d.upper()}/Recall-{rec:.3g}-{anno}-{d.upper()}-QPS-Comp.jpg", dpi=200)
      plt.close()


def draw_qps_comp_fixed_recall_by_selectivity(da, datasets, methods, anno, *, d_m_b={}, m_b_s={}, prefix="figures"):
  df = pd.read_csv(f"stats-{da}d.csv", dtype=types)
  recall_thresholds = [0.8, 0.9, 0.95]
  selectivities = da_sel[da]

  for rec in recall_thresholds:
    fig, axs = plt.subplots(2, len(datasets), layout='constrained')

    for i, d in enumerate(datasets):
      data = df[df["dataset"] == d]
      for m in methods:
        marker = m_marker[m]
        for b in d_m_b.get(d, {}).get(m, data[data["method"] == m].build.unique()):
          data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
          if m.startswith("Compass"):
            for nrel in compass_args["nrel"]:
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].str.contains(f"nrel_{nrel}")]
              rec_sel_qps_comp = data_by_m_b_nrel[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
              grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
              grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
              pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
              axs[0][i].plot(pos_s, grouped_qps["qps"])
              axs[0][i].scatter(pos_s, grouped_qps["qps"], label=f"{m}-{b}-{rec}-{nrel}", marker=marker)
              axs[1][i].plot(pos_s, grouped_comp["ncomp"])
              axs[1][i].scatter(pos_s, grouped_comp["ncomp"], label=f"{m}-{b}-{rec}-{nrel}", marker=marker)
          else:
            rec_sel_qps_comp = data_by_m_b[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
            grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
            grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
            pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
            axs[0][i].plot(pos_s, grouped_qps["qps"])
            axs[0][i].scatter(pos_s, grouped_qps["qps"], label=f"{m}-{b}-{rec}", marker=marker)
            axs[1][i].plot(pos_s, grouped_comp["ncomp"])
            axs[1][i].scatter(pos_s, grouped_comp["ncomp"], label=f"{m}-{b}-{rec}", marker=marker)

      axs[0][i].set_xticks(np.arange(len(selectivities)))
      axs[0][i].set_xticklabels(selectivities)
      axs[0][i].set_xlabel('Selectivity')
      axs[0][i].set_ylabel('QPS')
      axs[0][i].set_title(f"{d.upper()}, Recall-{rec:.3g}")
      axs[1][i].set_xticks(np.arange(len(selectivities)))
      axs[1][i].set_xticklabels(selectivities)
      axs[1][i].set_ylabel('# Comp')
      axs[1][i].set_xlabel('Selectivity')
      axs[1][i].set_title(f"{d.upper()}, Recall-{rec:.3g}")

    fig.set_size_inches(26, 7)
    unique_labels = {}
    for ax in axs.flat:
      handles, labels = ax.get_legend_handles_labels()
      for handle, label in zip(handles, labels):
        if label not in unique_labels:
          unique_labels[label] = handle
    fig.legend(unique_labels.values(), unique_labels.keys(), loc="outside right upper")
    fig.savefig(f"{prefix}{da}d-10/Recall-{rec:.3g}-{anno}-All-QPS-Comp.jpg", dpi=200)
    plt.close()


if __name__ == "__main__":
  summarize()
  for dim in [4]:
    draw_qps_comp_wrt_recall_by_dataset_selectivity(dim, DATASETS, METHODS)
    draw_qps_comp_wrt_recall_by_selectivity(dim, DATASETS, METHODS)
    draw_qps_comp_fixed_recall_by_dataset_selectivity(dim, DATASETS, METHODS, "MoM")
    draw_qps_comp_fixed_recall_by_selectivity(dim, DATASETS, METHODS, "MoM")
