import bisect
import json
from functools import reduce
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import (
  DA_RANGE,
  DA_S,
  DA_SEL,
  M_STYLE,
  compass_args,
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
  "tqps": float,
  "ncomp": float,
  "prop": float,
}

xlim = [0.8, 1]

def draw_qps_comp_wrt_recall_by_selectivity_camera(da, datasets, methods, anno, *, d_m_b={}, d_m_s={}, prefix="figures", ranges=[]):
  xlim = [0.8, 1]
  df = pd.read_csv(f"stats-{da}d.csv", dtype=types)
  df = df.fillna('')
  dataset_comp_ylim = {
    "crawl": 10000,
    "gist-dedup": 10000,
    "video-dedup": 30000,
    "glove100": 30000,
  }

  for rg in ranges if ranges else DA_RANGE[da]:
    fig, axs = plt.subplots(2, len(datasets), layout='constrained')
    for i, d in enumerate(datasets):
      selector = ((df["dataset"] == d) & (df["range"] == rg))
      if not selector.any():
        continue

      data = df[selector]
      sel = float(data["selectivity"].unique()[0])
      for m in d_m_b[d].keys() if d in d_m_b else methods:
        marker = M_STYLE[m]
        for b in d_m_b.get(d, {}).get(m, data[data["method"] == m].build.unique()):
          if da > 1 and (m == "SeRF" or m == "iRangeGraph"):
            data_by_m_b = data[(data["method"] == m + "+Post") & (data["build"] == b)]
          else:
            data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
          if m.startswith("Compass"):
            for nrel in d_m_s.get(d, {}).get(m, {}).get("nrel", compass_args["nrel"]):
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].str.contains(f"nrel_{nrel}")]
              rec_qps_comp = data_by_m_b_nrel[["recall", "qps", "ncomp", "initial_ncomp"]].sort_values(["recall"], ascending=[True])
              rec_qps_comp["total_ncomp"] = rec_qps_comp["initial_ncomp"] + rec_qps_comp["ncomp"]

              recall_above = rec_qps_comp[rec_qps_comp["recall"].gt(xlim[0])]
              axs[0][i].plot(recall_above["recall"], recall_above["qps"], **marker)
              axs[0][i].scatter(recall_above["recall"], recall_above["qps"], label=f"{m}-{b}-nrel_{nrel}", **marker)
              axs[1][i].plot(recall_above["recall"], recall_above["total_ncomp"], **marker)
              axs[1][i].scatter(recall_above["recall"], recall_above["total_ncomp"], label=f"{m}-{b}-nrel_{nrel}", **marker)
          else:
            recall_qps = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
            recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
            axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
            axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", **marker)

            recall_ncomp = data_by_m_b[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
            recall_ncomp = recall_ncomp[recall_ncomp["recall"].gt(xlim[0])].to_numpy()
            axs[1][i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1], **marker)
            axs[1][i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1], label=f"{m}-{b}", **marker)

          dt = d.split("-")[0].upper()
          axs[0][i].set_xlabel('Recall')
          axs[0][i].set_ylabel('QPS')
          axs[0][i].set_title("{}, Selectivity-{:.1%}".format(dt, sel))
          axs[1][i].set_xlabel('Recall')
          axs[1][i].set_ylabel('# Comp')
          auto_bottom, auto_top = axs[1][i].get_ylim()
          axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
          axs[1][i].set_title("{}, Selectivity-{:.1%}".format(dt, sel))

      fig.set_size_inches(10, 6)
      unique_labels = {}
      for ax in axs[0]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
          label = label.split("-")[0]
          if label.startswith("Compass"):
            label = "Compass"
          if label.startswith("SeRF"):
            label = "SeRF"
          if label.startswith("Navix"):
            label = "NaviX"
          if label not in unique_labels:
            unique_labels[label] = handle
        ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper right")
      for ax in axs[1]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
          label = label.split("-")[0]
          if label.startswith("Compass"):
            label = "Compass"
          if label.startswith("SeRF"):
            label = "SeRF"
          if label.startswith("Navix"):
            label = "NaviX"
          if label not in unique_labels:
            unique_labels[label] = handle
        ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper left")
      # fig.legend(unique_labels.values(), unique_labels.keys(), loc="upper right")
      # plt.grid(True)
      path = Path(f"{prefix}/All-{anno}-{rg}-QPS-Comp-Recall.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      fig.savefig(path, dpi=200)
      plt.close()

def draw_qps_comp_fixing_recall_by_selectivity_camera(da, datasets, methods, anno, *, d_m_b={}, d_m_s={}, prefix="figures"):
  df = pd.read_csv(f"stats-{da}d.csv", dtype=types)
  df = df.fillna('')
  recall_thresholds = [0.9, 0.95]
  selectivities = DA_SEL[da]
  dataset_comp_ylim = {
    "crawl": 10000,
    "gist-dedup": 10000,
    "video-dedup": 30000,
    "glove100": 30000,
  }

  for rec in recall_thresholds:
    fig, axs = plt.subplots(2, len(datasets), layout='constrained')

    for i, d in enumerate(datasets):
      data = df[df["dataset"] == d]
      for m in d_m_b[d].keys() if d in d_m_b else methods:
        marker = M_STYLE[m]
        for b in d_m_b.get(d, {}).get(m, data[data["method"] == m].build.unique()):
          if da > 1 and (m == "SeRF" or m == "iRangeGraph"):
            data_by_m_b = data[(data["method"] == m + "+Post") & (data["build"] == b)]
          else:
            data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
          if m.startswith("Compass"):
            for nrel in d_m_s.get(d, {}).get(m, {}).get("nrel", compass_args["nrel"]):
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].str.contains(f"nrel_{nrel}")]
              rec_sel_qps_ncomp = data_by_m_b_nrel[["recall", "selectivity", "qps", "ncomp", "initial_ncomp"]].sort_values(["selectivity", "recall"])
              rec_sel_qps_ncomp["total_ncomp"] = rec_sel_qps_ncomp["initial_ncomp"] + rec_sel_qps_ncomp["ncomp"]
              grouped_qps = rec_sel_qps_ncomp[rec_sel_qps_ncomp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
              grouped_ncomp = rec_sel_qps_ncomp[rec_sel_qps_ncomp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
              grouped_total_ncomp = rec_sel_qps_ncomp[rec_sel_qps_ncomp["recall"].gt(rec)].groupby("selectivity", as_index=False)["total_ncomp"].min()
              pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
              p = axs[0][i].plot(pos_s, grouped_qps["qps"], **marker)
              axs[0][i].scatter(pos_s, grouped_qps["qps"], label=f"{m}-{b}-{rec}-{nrel}", **marker)
              # axs[1][i].plot(pos_s, grouped_ncomp["ncomp"])
              # axs[1][i].scatter(pos_s, grouped_ncomp["ncomp"], label=f"{m}-{b}-{rec}-{nrel}", **marker)
              axs[1][i].plot(pos_s, grouped_total_ncomp["total_ncomp"], color=p[0].get_color())
              axs[1][i].scatter(pos_s, grouped_total_ncomp["total_ncomp"], label=f"{m}-{b}-{rec}-{nrel}", **marker)
          else:
            rec_sel_qps_ncomp = data_by_m_b[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
            grouped_qps = rec_sel_qps_ncomp[rec_sel_qps_ncomp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
            grouped_ncomp = rec_sel_qps_ncomp[rec_sel_qps_ncomp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
            pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
            axs[0][i].plot(pos_s, grouped_qps["qps"], **marker)
            axs[0][i].scatter(pos_s, grouped_qps["qps"], label=f"{m}-{b}-{rec}", **marker)
            axs[1][i].plot(pos_s, grouped_ncomp["ncomp"], **marker)
            axs[1][i].scatter(pos_s, grouped_ncomp["ncomp"], label=f"{m}-{b}-{rec}", **marker)

      dt = d.split("-")[0].upper()
      axs[0][i].set_xticks(np.arange(len(selectivities)))
      axs[0][i].set_xticklabels(selectivities)
      axs[0][i].set_xlabel('Selectivity')
      axs[0][i].set_ylabel('QPS')
      axs[0][i].set_title(f"{dt}, Recall-{rec:.3g}")
      axs[1][i].set_xticks(np.arange(len(selectivities)))
      axs[1][i].set_xticklabels(selectivities)
      axs[1][i].set_xlabel('Selectivity')
      axs[1][i].set_ylabel('# Comp')
      auto_bottom, auto_top = axs[1][i].get_ylim()
      axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
      axs[1][i].set_title(f"{dt}, Recall-{rec:.3g}")

    fig.set_size_inches(12, 6)
    unique_labels = {}
    for ax in axs.flat:
      handles, labels = ax.get_legend_handles_labels()
      for handle, label in zip(handles, labels):
        label = label.split("-")[0]
        if label.startswith("Compass"):
          label = "Compass"
        if label.startswith("SeRF"):
          label = "SeRF"
        if label.startswith("Navix"):
          label = "NaviX"
        if label not in unique_labels:
          unique_labels[label] = handle
      ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper right")
    path = Path(f"{prefix}/Recall-{rec:.3g}-{anno}-All-QPS-Comp.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    # plt.grid(True)
    fig.savefig(path, dpi=200)
    plt.close()

def draw_qps_comp_fixing_dimension_selectivity_by_dimension_camera(datasets, d_m_b, d_m_s, anno, prefix):
  sel_s = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  interval = {
    0.3: ["0.3", "0.09", "0.027", "0.0081"],
    0.4: ["0.4", "0.16", "0.064", "0.0256"],
    0.5: ["0.5", "0.25", "0.125", "0.0625"],
    0.6: ["0.6", "0.36", "0.216", "0.1296"],
    0.7: ["0.7", "0.49", "0.343", "0.2401"],
    0.8: ["0.8", "0.64", "0.512", "0.6561"],
    0.9: ["0.9", "0.81", "0.729", "0.8561"],
  }
  dataset_comp_ylim = {
    "crawl": 15000,
    "gist-dedup": 15000,
    "video-dedup": 35000,
    "glove100": 30000,
  }

  for rec in [0.8, 0.85, 0.9, 0.95]:
    d_m_sel = {}
    for d in d_m_b.keys():
      d_m_sel[d] = {}
      for m in d_m_b[d].keys():
        d_m_sel[d][m] = {}
        for sel in sel_s:
          d_m_sel[d][m][sel] = {}

    for da in DA_S:
      df = pd.read_csv(f"stats-{da}d.csv", dtype=types)
      for i, d in enumerate(datasets):
        data = df[df["dataset"] == d]
        for sel in sel_s:
          for m in d_m_b[d].keys():
            for b in d_m_b[d][m]:
              if da > 1 and (m == "SeRF" or m == "iRangeGraph"):
                data_by_m_b = data[(data["method"] == m + "+Post") & (data["build"] == b)]
              else:
                data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
              if m.startswith("Compass"):
                for nrel in d_m_s[d][m]["nrel"]:
                  data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].str.contains(f"nrel_{nrel}")]
                  rec_sel_qps_comp = data_by_m_b_nrel[["recall", "selectivity", "qps", "ncomp",
                                                        "initial_ncomp"]].sort_values(["selectivity", "recall"])
                  rec_sel_qps_comp["total_ncomp"] = rec_sel_qps_comp["initial_ncomp"] + rec_sel_qps_comp["ncomp"]
                  grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
                  grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
                  grouped_total_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity",
                                                                                                    as_index=False)["total_ncomp"].min()
                  pos = bisect.bisect(grouped_qps["selectivity"], interval[sel][da - 1]) - 1
                  if pos == -1 or grouped_qps["selectivity"][pos] != interval[sel][da - 1]:
                    pos = -1
                  label = f"{m}-{b}-{nrel}-{rec}"
                  if label not in d_m_sel[d][m][sel]:  # a list by dimension
                    d_m_sel[d][m][sel][label] = {"qps": [], "ncomp": [], "total_ncomp": []}
                  d_m_sel[d][m][sel][label]["qps"].append(grouped_qps["qps"][pos] if pos >= 0 else 0)
                  d_m_sel[d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else 30000)
                  d_m_sel[d][m][sel][label]["total_ncomp"].append(grouped_total_comp["total_ncomp"][pos] if pos >= 0 else 30000)
              else:
                rec_sel_qps_comp = data_by_m_b[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
                grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
                grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
                pos = bisect.bisect(grouped_qps["selectivity"], interval[sel][da - 1]) - 1
                if pos == -1 or grouped_qps["selectivity"][pos] != interval[sel][da - 1]:
                  pos = -1
                  fallout = rec_sel_qps_comp[rec_sel_qps_comp["selectivity"] == interval[sel][da - 1]]["ncomp"].max()
                label = f"{m}-{b}-{rec}"
                if label not in d_m_sel[d][m][sel]:
                  d_m_sel[d][m][sel][label] = {"qps": [], "ncomp": []}
                d_m_sel[d][m][sel][label]["qps"].append(grouped_qps["qps"][pos] if pos >= 0 else 0)
                d_m_sel[d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else fallout)

    for sel in sel_s:
      fig, axs = plt.subplots(2, len(datasets), layout='constrained')
      for i, d in enumerate(datasets):
        for m in d_m_b[d].keys():
          marker = M_STYLE[m]
          for label in d_m_sel[d][m][sel].keys():
            das = DA_S[:len(d_m_sel[d][m][sel][label]["qps"])]
            sc = axs[0][i].scatter(das, d_m_sel[d][m][sel][label]["qps"], label=label, **marker)
            axs[0][i].plot(das, d_m_sel[d][m][sel][label]["qps"], color=sc.get_facecolor()[0])
            if m.startswith("Compass"):
              axs[1][i].scatter(das, d_m_sel[d][m][sel][label]["total_ncomp"], label=label, **marker)
              axs[1][i].plot(das, d_m_sel[d][m][sel][label]["total_ncomp"], color=sc.get_facecolor()[0])
            else:
              axs[1][i].scatter(das, d_m_sel[d][m][sel][label]["ncomp"], label=label, **marker)
              axs[1][i].plot(das, d_m_sel[d][m][sel][label]["ncomp"], color=sc.get_facecolor()[0])

        dt = d.split("-")[0].upper()
        axs[0][i].set_xlabel('Dimension')
        axs[0][i].set_xticks(DA_S)
        axs[0][i].set_ylabel('QPS')
        axs[0][i].set_title(f"{dt}, Recall-{rec:.3g}")
        axs[1][i].set_xlabel('Dimension')
        axs[1][i].set_ylabel('# Comp')
        auto_bottom, auto_top = axs[1][i].get_ylim()
        axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
        axs[1][i].set_xticks(DA_S)
        axs[1][i].set_title(f"{dt}, Recall-{rec:.3g}")

      fig.set_size_inches(12, 6)
      unique_labels = {}
      for ax in axs[0]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
          label = label.split("-")[0]
          if label.startswith("Compass"):
            label = "Compass"
          if label.startswith("SeRF"):
            label = "SeRF"
          if label.startswith("Navix"):
            label = "NaviX"
          if label not in unique_labels:
            unique_labels[label] = handle
        ax.legend(unique_labels.values(), unique_labels.keys(), loc="best")
      for ax in axs[1]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
          label = label.split("-")[0]
          if label.startswith("Compass"):
            label = "Compass"
          if label.startswith("SeRF"):
            label = "SeRF"
          if label.startswith("Navix"):
            label = "NaviX"
          if label not in unique_labels:
            unique_labels[label] = handle
        ax.legend(unique_labels.values(), unique_labels.keys(), loc="best")
      # fig.legend(unique_labels.values(), unique_labels.keys(), loc="upper right")
      path = Path(f"{prefix}/Sel-{sel:.3g}-Recall-{rec:.3g}-{anno}-All-QPS-Comp.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      # plt.grid(True)
      fig.savefig(path, dpi=200)
      plt.close()


def draw_qps_comp_with_disjunction_by_dimension_camera(datasets, d_m_b, d_m_s, anno, prefix):
  sel_s = [0.1, 0.2, 0.3]
  interval = {
    0.1: ["0.1", "0.2", "0.3", "0.4"],
    0.2: ["0.2", "0.4", "0.6", "0.8"],
    0.3: ["0.3", "0.6", "0.9", "1"],
  }
  dataset_comp_ylim = {
    "crawl": 10000,
    "gist-dedup": 10000,
    "video-dedup": 30000,
    "glove100": 30000,
  }

  ndisjunctions = [1, 2, 3, 4]

  for rec in [0.8, 0.85, 0.9, 0.95]:
    d_m = {}
    for d in d_m_b.keys():
      d_m[d] = {}
      for m in d_m_b[d].keys():
        d_m[d][m] = {}
        for sel in sel_s:
          d_m[d][m][sel] = {}

    for ndis in ndisjunctions:
      dataset_comp_ylim_4d = {
        "crawl": 10000,
        "gist-dedup": 10000,
        "video-dedup": 80000,
        "glove100": 80000,
      }
      if ndis == 4: dataset_comp_ylim = dataset_comp_ylim_4d
      df = pd.read_csv("stats-1d.csv", dtype=types)
      for i, d in enumerate(datasets):
        data = df[df["dataset"] == d]
        for sel in sel_s:
          for m in d_m_b[d].keys():
            for b in d_m_b[d][m]:
              if m == "SeRF":
                if ndis == 1:
                  data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
                elif ndis == 2:
                  data_by_m_b = data[(data["method"] == m + "+OR") & (data["build"] == b)]
                elif ndis == 3:
                  data_by_m_b = data[(data["method"] == m + "+OR3") & (data["build"] == b)]
                elif ndis == 4:
                  data_by_m_b = data[(data["method"] == m + "+OR3") & (data["build"] == b)]
              else:
                data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
              if m.startswith("Compass"):
                for nrel in d_m_s[d][m]["nrel"]:
                  data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].str.contains(f"nrel_{nrel}")]
                  rec_sel_qps_comp = data_by_m_b_nrel[["recall", "selectivity", "qps", "ncomp",
                                                        "initial_ncomp"]].sort_values(["selectivity", "recall"])
                  rec_sel_qps_comp["total_ncomp"] = rec_sel_qps_comp["initial_ncomp"] + rec_sel_qps_comp["ncomp"]
                  grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
                  grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
                  grouped_total_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity",
                                                                                                    as_index=False)["total_ncomp"].min()
                  pos = bisect.bisect(grouped_qps["selectivity"], interval[sel][ndis - 1]) - 1
                  if pos == -1 or grouped_qps["selectivity"][pos] != interval[sel][ndis - 1]:
                    pos = -1
                  label = f"{m}-{b}-{nrel}-{rec}"
                  if label not in d_m[d][m][sel]:  # a list by dimension
                    d_m[d][m][sel][label] = {"qps": [], "ncomp": [], "total_ncomp": []}
                  d_m[d][m][sel][label]["qps"].append(grouped_qps["qps"][pos] if pos >= 0 else 0)
                  d_m[d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else 30000)
                  d_m[d][m][sel][label]["total_ncomp"].append(grouped_total_comp["total_ncomp"][pos] if pos >= 0 else 30000)
              else:
                rec_sel_qps_comp = data_by_m_b[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
                grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
                grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
                pos = bisect.bisect(grouped_qps["selectivity"], interval[sel][ndis - 1]) - 1
                if pos == -1 or grouped_qps["selectivity"][pos] != interval[sel][ndis - 1]:
                  pos = -1
                  fallout = rec_sel_qps_comp[rec_sel_qps_comp["selectivity"] == interval[sel][ndis - 1]]["ncomp"].max()
                label = f"{m}-{b}-{rec}"
                if label not in d_m[d][m][sel]:
                  d_m[d][m][sel][label] = {"qps": [], "ncomp": []}
                d_m[d][m][sel][label]["qps"].append(grouped_qps["qps"][pos] if pos >= 0 else 0)
                d_m[d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else fallout)

    for sel in sel_s:
      fig, axs = plt.subplots(2, len(datasets), layout='constrained')
      for i, d in enumerate(datasets):
        for m in d_m_b[d].keys():
          marker = M_STYLE[m]
          for label in d_m[d][m][sel].keys():
            sc = axs[0][i].scatter(ndisjunctions, d_m[d][m][sel][label]["qps"], label=label, **marker)
            axs[0][i].plot(ndisjunctions, d_m[d][m][sel][label]["qps"], color=sc.get_facecolor()[0])
            if m.startswith("Compass"):
              axs[1][i].scatter(ndisjunctions, d_m[d][m][sel][label]["total_ncomp"], label=label, **marker)
              axs[1][i].plot(ndisjunctions, d_m[d][m][sel][label]["total_ncomp"], color=sc.get_facecolor()[0])
            else:
              axs[1][i].scatter(ndisjunctions, d_m[d][m][sel][label]["ncomp"], label=label, **marker)
              axs[1][i].plot(ndisjunctions, d_m[d][m][sel][label]["ncomp"], color=sc.get_facecolor()[0])

        dt = d.split("-")[0].upper()
        axs[0][i].set_xlabel('Dimension')
        axs[0][i].set_xticks(ndisjunctions)
        axs[0][i].set_ylabel('QPS')
        axs[0][i].set_title(f"{dt}, Recall-{rec:.3g}")
        axs[1][i].set_xlabel('Dimension')
        axs[1][i].set_ylabel('# Comp')
        auto_bottom, auto_top = axs[1][i].get_ylim()
        axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
        axs[1][i].set_xticks(ndisjunctions)
        axs[1][i].set_title(f"{dt}, Recall-{rec:.3g}")

      fig.set_size_inches(12, 6)
      unique_labels = {}
      for ax in axs[0]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
          label = label.split("-")[0]
          if label.startswith("Compass"):
            label = "Compass"
          if label.startswith("SeRF"):
            label = "SeRF"
          if label.startswith("Navix"):
            label = "NaviX"
          if label not in unique_labels:
            unique_labels[label] = handle
        ax.legend(unique_labels.values(), unique_labels.keys(), loc="best")
      for ax in axs[1]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
          label = label.split("-")[0]
          if label.startswith("Compass"):
            label = "Compass"
          if label.startswith("SeRF"):
            label = "SeRF"
          if label.startswith("Navix"):
            label = "NaviX"
          if label not in unique_labels:
            unique_labels[label] = handle
        ax.legend(unique_labels.values(), unique_labels.keys(), loc="best")
      # fig.legend(unique_labels.values(), unique_labels.keys(), loc="upper right")
      path = Path(f"{prefix}/Sel-{sel:.3g}-Recall-{rec:.3g}-{anno}-All-QPS-Comp.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      # plt.grid(True)
      fig.savefig(path, dpi=200)
      plt.close()

