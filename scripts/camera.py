import bisect
import json
from functools import reduce
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import (
  D_ARGS,
  DA_RANGE,
  DA_S,
  DA_SEL,
  M_ARGS,
  M_DA_RUN,
  M_PARAM,
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
  xlim, xticks = [0.8, 1], [0.8, 0.9, 1]
  if ranges:  # ablations study
    xlim, xticks = [0.4, 1], [0.4, 0.6, 0.8, 1]
  df = pd.read_csv(f"stats-{da}d.csv", dtype=types)
  df = df.fillna('')
  dataset_comp_ylim = {
    "crawl": 10000,
    "gist-dedup": 10000,
    "video-dedup": 30000,
    "glove100": 30000,
  }

  selected_efs = [10, 20, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 500, 600, 800, 1000]

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
              selected_search = [f"efs_{efs}_nrel_{nrel}_batch_k_20_initial_efs_20_delta_efs_20" for efs in selected_efs]
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].isin(selected_search)]
              rec_qps_comp = data_by_m_b_nrel[["recall", "qps", "ncomp", "initial_ncomp"]].sort_values(["recall"], ascending=[True])
              rec_qps_comp["total_ncomp"] = rec_qps_comp["initial_ncomp"] + rec_qps_comp["ncomp"]

              recall_above = rec_qps_comp[rec_qps_comp["recall"].gt(xlim[0])]
              axs[0][i].plot(recall_above["recall"], recall_above["qps"], **marker)
              axs[0][i].scatter(recall_above["recall"], recall_above["qps"], label=f"{m}-{b}-nrel_{nrel}", **marker)
              axs[1][i].plot(recall_above["recall"], recall_above["total_ncomp"], **marker)
              axs[1][i].scatter(recall_above["recall"], recall_above["total_ncomp"], label=f"{m}-{b}-nrel_{nrel}", **marker)
          else:
            selected_search = [f"efs_{efs}" for efs in selected_efs]
            data_by_m_b = data_by_m_b[data_by_m_b["search"].isin(selected_search)]
            recall_qps = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
            recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
            axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
            axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", **marker)

            recall_ncomp = data_by_m_b[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
            recall_ncomp = recall_ncomp[recall_ncomp["recall"].gt(xlim[0])].to_numpy()
            axs[1][i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1], **marker)
            axs[1][i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1], label=f"{m}-{b}", **marker)

          dt = d.split("-")[0].upper()
          # axs[0][i].set_xlabel('Recall')
          axs[0][i].set_xticks(xticks)
          if i == 0:
            axs[0][i].set_ylabel('QPS')
          axs[0][i].set_title("{}, Passrate-{:.1%}".format(dt, sel))
          axs[1][i].set_xlabel('Recall')
          axs[1][i].set_xticks(xticks)
          if i == 0:
            axs[1][i].set_ylabel('# Comp')
          auto_bottom, auto_top = axs[1][i].get_ylim()
          axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
          # axs[1][i].set_title("{}, Passrate-{:.1%}".format(dt, sel))

      fig.set_size_inches(10, 6)
      unique_labels = {}
      for ax in axs[0]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
          label = label.split("-")[0]
          if label.startswith("Compass"):
            label = label if label != "CompassPostKTh" else "Compass"
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
            label = label if label != "CompassPostKTh" else "Compass"
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
      plt.close("all")


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
      # axs[0][i].set_xlabel('Passrate')
      if i == 0:
        axs[0][i].set_ylabel('QPS')
      axs[0][i].set_title(f"{dt}, Recall-{rec:.3g}")
      axs[1][i].set_xticks(np.arange(len(selectivities)))
      axs[1][i].set_xticklabels(selectivities)
      axs[1][i].set_xlabel('Passrate')
      if i == 0:
        axs[1][i].set_ylabel('# Comp')
      auto_bottom, auto_top = axs[1][i].get_ylim()
      axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
      # axs[1][i].set_title(f"{dt}, Recall-{rec:.3g}")

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
    plt.close("all")


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

  rec_d_m_sel = {}
  for rec in [0.8, 0.85, 0.9, 0.95]:
    rec_d_m_sel[rec] = {}
    for d in d_m_b.keys():
      rec_d_m_sel[rec][d] = {}
      for m in d_m_b[d].keys():
        rec_d_m_sel[rec][d][m] = {}
        for sel in sel_s:
          rec_d_m_sel[rec][d][m][sel] = {}

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
                  if label not in rec_d_m_sel[rec][d][m][sel]:  # a list by dimension
                    rec_d_m_sel[rec][d][m][sel][label] = {"qps": [], "ncomp": [], "total_ncomp": []}
                  rec_d_m_sel[rec][d][m][sel][label]["qps"].append(grouped_qps["qps"][pos] if pos >= 0 else 0)
                  rec_d_m_sel[rec][d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else 30000)
                  rec_d_m_sel[rec][d][m][sel][label]["total_ncomp"].append(grouped_total_comp["total_ncomp"][pos] if pos >= 0 else 30000)
              else:
                rec_sel_qps_comp = data_by_m_b[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
                grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
                grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
                pos = bisect.bisect(grouped_qps["selectivity"], interval[sel][da - 1]) - 1
                if pos == -1 or grouped_qps["selectivity"][pos] != interval[sel][da - 1]:
                  pos = -1
                  fallout_qps = rec_sel_qps_comp[rec_sel_qps_comp["selectivity"] == interval[sel][da - 1]]["qps"].min()
                  fallout_ncomp = rec_sel_qps_comp[rec_sel_qps_comp["selectivity"] == interval[sel][da - 1]]["ncomp"].max()
                label = f"{m}-{b}-{rec}"
                if label not in rec_d_m_sel[rec][d][m][sel]:
                  rec_d_m_sel[rec][d][m][sel][label] = {"qps": [], "ncomp": [], "fallout": []}
                rec_d_m_sel[rec][d][m][sel][label]["qps"].append(grouped_qps["qps"][pos] if pos >= 0 else fallout_qps)
                rec_d_m_sel[rec][d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else fallout_ncomp)
                if pos == -1:
                  rec_d_m_sel[rec][d][m][sel][label]["fallout"].append(len(rec_d_m_sel[rec][d][m][sel][label]["qps"]))

  for rec in [0.8, 0.85, 0.9, 0.95]:
    for sel in sel_s:
      fig, axs = plt.subplots(2, len(datasets), layout='constrained')
      for i, d in enumerate(datasets):
        for m in d_m_b[d].keys():
          marker = M_STYLE[m]
          for label in rec_d_m_sel[rec][d][m][sel].keys():
            das = DA_S[:len(rec_d_m_sel[rec][d][m][sel][label]["qps"])]
            sc = axs[0][i].scatter(das, rec_d_m_sel[rec][d][m][sel][label]["qps"], label=label, **marker)
            axs[0][i].plot(das, rec_d_m_sel[rec][d][m][sel][label]["qps"], color=sc.get_facecolor()[0])
            if m.startswith("Compass"):
              axs[1][i].scatter(das, rec_d_m_sel[rec][d][m][sel][label]["total_ncomp"], label=label, **marker)
              axs[1][i].plot(das, rec_d_m_sel[rec][d][m][sel][label]["total_ncomp"], color=sc.get_facecolor()[0])
            else:
              axs[1][i].scatter(das, rec_d_m_sel[rec][d][m][sel][label]["ncomp"], label=label, **marker)
              axs[1][i].plot(das, rec_d_m_sel[rec][d][m][sel][label]["ncomp"], color=sc.get_facecolor()[0])
            for tick in rec_d_m_sel[rec][d][m][sel][label].get("fallout", []):
              # mark a cross sign at (tick, fallout_qps)
              axs[0][i].scatter(tick, rec_d_m_sel[rec][d][m][sel][label]["qps"][tick - 1], s=100, color=sc.get_facecolor()[0], marker='x')
              axs[1][i].scatter(tick, rec_d_m_sel[rec][d][m][sel][label]["ncomp"][tick - 1], s=100, color=sc.get_facecolor()[0], marker='x')
        dt = d.split("-")[0].upper()
        # axs[0][i].set_xlabel('Dimension')
        axs[0][i].set_xticks(DA_S)
        if i == 0:
          axs[0][i].set_ylabel('QPS')
        axs[0][i].set_title(f"{dt}, Recall-{rec:.3g}")
        axs[1][i].set_xlabel('Dimension')
        if i == 0:
          axs[1][i].set_ylabel('# Comp')
        auto_bottom, auto_top = axs[1][i].get_ylim()
        axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
        axs[1][i].set_xticks(DA_S)
        # axs[1][i].set_title(f"{dt}, Recall-{rec:.3g}")

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
      path = Path(f"{prefix}/Shrinked-Sel-{sel:.3g}-Recall-{rec:.3g}-{anno}-All-QPS-Comp.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      # plt.grid(True)
      fig.savefig(path, dpi=200)
      plt.close("all")

  for rec_s, dataset_s in zip([(0.85, 0.95)], [("video-dedup", "gist-dedup")]):
    for sel in sel_s:
      fig, axs = plt.subplots(2, len(datasets), layout='constrained')
      for rec_i, rec in enumerate(rec_s):
        for d_i, d in enumerate(dataset_s):
          i = rec_i * len(dataset_s) + d_i
          for m in d_m_b[d].keys():
            marker = M_STYLE[m]
            for label in rec_d_m_sel[rec][d][m][sel].keys():
              das = DA_S[:len(rec_d_m_sel[rec][d][m][sel][label]["qps"])]
              sc = axs[0][i].scatter(das, rec_d_m_sel[rec][d][m][sel][label]["qps"], label=label, **marker)
              axs[0][i].plot(das, rec_d_m_sel[rec][d][m][sel][label]["qps"], color=sc.get_facecolor()[0])
              if m.startswith("Compass"):
                axs[1][i].scatter(das, rec_d_m_sel[rec][d][m][sel][label]["total_ncomp"], label=label, **marker)
                axs[1][i].plot(das, rec_d_m_sel[rec][d][m][sel][label]["total_ncomp"], color=sc.get_facecolor()[0])
              else:
                axs[1][i].scatter(das, rec_d_m_sel[rec][d][m][sel][label]["ncomp"], label=label, **marker)
                axs[1][i].plot(das, rec_d_m_sel[rec][d][m][sel][label]["ncomp"], color=sc.get_facecolor()[0])
              for tick in rec_d_m_sel[rec][d][m][sel][label].get("fallout", []):
                # mark a cross sign at (tick, fallout_qps)
                axs[0][i].scatter(tick, rec_d_m_sel[rec][d][m][sel][label]["qps"][tick - 1], s=100, color=sc.get_facecolor()[0], marker='x')
                axs[1][i].scatter(tick, rec_d_m_sel[rec][d][m][sel][label]["ncomp"][tick - 1], s=100, color=sc.get_facecolor()[0], marker='x')
          dt = d.split("-")[0].upper()
          # axs[0][i].set_xlabel('Dimension')
          axs[0][i].set_xticks(DA_S)
          if i == 0:
            axs[0][i].set_ylabel('QPS')
          axs[0][i].set_title(f"{dt}, Recall-{rec:.3g}")
          axs[1][i].set_xlabel('Dimension')
          if i == 0:
            axs[1][i].set_ylabel('# Comp')
          auto_bottom, auto_top = axs[1][i].get_ylim()
          axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
          axs[1][i].set_xticks(DA_S)
          # axs[1][i].set_title(f"{dt}, Recall-{rec:.3g}")

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
      path = Path(f"{prefix}/Sel-{sel:.3g}-Recall-{rec_s[0]:.3g}-{rec_s[1]:.3g}-{anno}-All-QPS-Comp.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      # plt.grid(True)
      fig.savefig(path, dpi=200)
      plt.close("all")


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

  rec_d_m = {}
  for rec in [0.8, 0.85, 0.9, 0.95]:
    rec_d_m[rec] = {}
    for d in d_m_b.keys():
      rec_d_m[rec][d] = {}
      for m in d_m_b[d].keys():
        rec_d_m[rec][d][m] = {}
        for sel in sel_s:
          rec_d_m[rec][d][m][sel] = {}

    for ndis in ndisjunctions:
      dataset_comp_ylim_4d = {
        "crawl": 10000,
        "gist-dedup": 10000,
        "video-dedup": 80000,
        "glove100": 80000,
      }
      if ndis == 4: dataset_comp_ylim = dataset_comp_ylim_4d  # noqa: E701
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
                  data_by_m_b = data[(data["method"] == m + "+OR4") & (data["build"] == b)]
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
                  if label not in rec_d_m[rec][d][m][sel]:  # a list by dimension
                    rec_d_m[rec][d][m][sel][label] = {"qps": [], "ncomp": [], "total_ncomp": []}
                  rec_d_m[rec][d][m][sel][label]["qps"].append(grouped_qps["qps"][pos] if pos >= 0 else 0)
                  rec_d_m[rec][d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else 30000)
                  rec_d_m[rec][d][m][sel][label]["total_ncomp"].append(grouped_total_comp["total_ncomp"][pos] if pos >= 0 else 30000)
              else:
                rec_sel_qps_comp = data_by_m_b[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
                grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
                grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
                pos = bisect.bisect(grouped_qps["selectivity"], interval[sel][ndis - 1]) - 1
                if pos == -1 or grouped_qps["selectivity"][pos] != interval[sel][ndis - 1]:
                  pos = -1
                  fallout_qps = rec_sel_qps_comp[rec_sel_qps_comp["selectivity"] == interval[sel][ndis - 1]]["qps"].min()
                  fallout_ncomp = rec_sel_qps_comp[rec_sel_qps_comp["selectivity"] == interval[sel][ndis - 1]]["ncomp"].max()
                label = f"{m}-{b}-{rec}"
                if label not in rec_d_m[rec][d][m][sel]:
                  rec_d_m[rec][d][m][sel][label] = {"qps": [], "ncomp": [], "fallout": []}
                rec_d_m[rec][d][m][sel][label]["qps"].append(grouped_qps["qps"][pos] if pos >= 0 else fallout_qps)
                rec_d_m[rec][d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else fallout_ncomp)
                if pos == -1:
                  rec_d_m[rec][d][m][sel][label]["fallout"].append(len(rec_d_m[rec][d][m][sel][label]["qps"]))

  for rec in [0.8, 0.85, 0.9, 0.95]:
    for sel in sel_s:
      fig, axs = plt.subplots(2, len(datasets), layout='constrained')
      for i, d in enumerate(datasets):
        for m in d_m_b[d].keys():
          marker = M_STYLE[m]
          for label in rec_d_m[rec][d][m][sel].keys():
            sc = axs[0][i].scatter(ndisjunctions, rec_d_m[rec][d][m][sel][label]["qps"], label=label, **marker)
            axs[0][i].plot(ndisjunctions, rec_d_m[rec][d][m][sel][label]["qps"], color=sc.get_facecolor()[0])
            if m.startswith("Compass"):
              axs[1][i].scatter(ndisjunctions, rec_d_m[rec][d][m][sel][label]["total_ncomp"], label=label, **marker)
              axs[1][i].plot(ndisjunctions, rec_d_m[rec][d][m][sel][label]["total_ncomp"], color=sc.get_facecolor()[0])
            else:
              axs[1][i].scatter(ndisjunctions, rec_d_m[rec][d][m][sel][label]["ncomp"], label=label, **marker)
              axs[1][i].plot(ndisjunctions, rec_d_m[rec][d][m][sel][label]["ncomp"], color=sc.get_facecolor()[0])
            for tick in rec_d_m[rec][d][m][sel][label].get("fallout", []):
              # mark a cross sign at (tick, fallout_qps)
              axs[0][i].scatter(tick, rec_d_m[rec][d][m][sel][label]["qps"][tick - 1], s=100, color=sc.get_facecolor()[0], marker='x')
              axs[1][i].scatter(tick, rec_d_m[rec][d][m][sel][label]["ncomp"][tick - 1], s=100, color=sc.get_facecolor()[0], marker='x')

        dt = d.split("-")[0].upper()
        # axs[0][i].set_xlabel('Dimension')
        axs[0][i].set_xticks(ndisjunctions)
        if i == 0:
          axs[0][i].set_ylabel('QPS')
        axs[0][i].set_title(f"{dt}, Recall-{rec:.3g}")
        axs[1][i].set_xlabel('Dimension')
        if i == 0:
          axs[1][i].set_ylabel('# Comp')
        auto_bottom, auto_top = axs[1][i].get_ylim()
        axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
        axs[1][i].set_xticks(ndisjunctions)
        # axs[1][i].set_title(f"{dt}, Recall-{rec:.3g}")

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
      plt.close("all")

  for rec_s, dataset_s in zip([(0.85, 0.95)], [("video-dedup", "gist-dedup")]):
    for sel in sel_s:
      fig, axs = plt.subplots(2, len(datasets), layout='constrained')
      for rec_i, rec in enumerate(rec_s):
        for d_i, d in enumerate(dataset_s):
          i = rec_i * len(dataset_s) + d_i
          for m in d_m_b[d].keys():
            marker = M_STYLE[m]
            for label in rec_d_m[rec][d][m][sel].keys():
              sc = axs[0][i].scatter(ndisjunctions, rec_d_m[rec][d][m][sel][label]["qps"], label=label, **marker)
              axs[0][i].plot(ndisjunctions, rec_d_m[rec][d][m][sel][label]["qps"], color=sc.get_facecolor()[0])
              if m.startswith("Compass"):
                axs[1][i].scatter(ndisjunctions, rec_d_m[rec][d][m][sel][label]["total_ncomp"], label=label, **marker)
                axs[1][i].plot(ndisjunctions, rec_d_m[rec][d][m][sel][label]["total_ncomp"], color=sc.get_facecolor()[0])
              else:
                axs[1][i].scatter(ndisjunctions, rec_d_m[rec][d][m][sel][label]["ncomp"], label=label, **marker)
                axs[1][i].plot(ndisjunctions, rec_d_m[rec][d][m][sel][label]["ncomp"], color=sc.get_facecolor()[0])
              for tick in rec_d_m[rec][d][m][sel][label].get("fallout", []):
                # mark a cross sign at (tick, fallout_qps)
                axs[0][i].scatter(tick, rec_d_m[rec][d][m][sel][label]["qps"][tick - 1], s=100, color=sc.get_facecolor()[0], marker='x')
                axs[1][i].scatter(tick, rec_d_m[rec][d][m][sel][label]["ncomp"][tick - 1], s=100, color=sc.get_facecolor()[0], marker='x')

          dt = d.split("-")[0].upper()
          # axs[0][i].set_xlabel('Dimension')
          axs[0][i].set_xticks(ndisjunctions)
          if i == 0:
            axs[0][i].set_ylabel('QPS')
          axs[0][i].set_title(f"{dt}, Recall-{rec:.3g}")
          axs[1][i].set_xlabel('Dimension')
          if i == 0:
            axs[1][i].set_ylabel('# Comp')
          auto_bottom, auto_top = axs[1][i].get_ylim()
          axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
          axs[1][i].set_xticks(ndisjunctions)
          # axs[1][i].set_title(f"{dt}, Recall-{rec:.3g}")

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
      path = Path(f"{prefix}/Sel-{sel:.3g}-Recall-{rec_s[0]:.3g}-{rec_s[1]:.3g}-{anno}-All-QPS-Comp.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      # plt.grid(True)
      fig.savefig(path, dpi=200)
      plt.close("all")


def summarize_multik(datasets):
  LOG_ROOT_MULTIK = "/opt/nfs_dcc/chunxy/logs_{}"
  m_s = ["CompassPostKTh", "Navix", "SeRF"]
  da = 1
  entries = []
  for m in m_s:
    for k in [5, 10, 15, 20, 25, 30]:
      if da not in M_DA_RUN[m]: continue  # noqa: E701
      for d in datasets:
        for itvl in M_DA_RUN[m][da]:
          if m == "CompassPostKTh" or m == "Navix":
            w = "{}_10000_{}_{}_{}".format(d, *map(lambda ele: "-".join(map(str, ele)), itvl), k)
            nrg = "-".join([f"{(r - l) // 100}" for l, r in zip(*itvl)])  # noqa: E741
            sel = f"{reduce(lambda a, b: a * b, [(r - l) / 10000 for l, r in zip(*itvl)], 1.):.4g}"  # noqa: E741
          elif m == "SeRF":
            w = "{}_{}_{}".format(d, "-".join(map(str, itvl)), k)
            nrg = "-".join(map(str, itvl))
            sel = f"{reduce(lambda a, b: a * b, map(lambda x: x / 100, itvl), 1.):.4g}"

          bt = "_".join([f"{bp}_{{}}" for bp in M_PARAM[m]["build"]])
          st = "_".join([f"{sp}_{{}}" for sp in M_PARAM[m]["search"]])
          if m.startswith("Compass"):
            ba_s = [D_ARGS[d].get(bp, M_ARGS[m][bp]) for bp in M_PARAM[m]["build"]]
            sa_s = [D_ARGS[d].get(sp, M_ARGS[m][sp]) for sp in M_PARAM[m]["search"]]
          else:
            ba_s = [M_ARGS[m][bp] for bp in M_PARAM[m]["build"]]
            sa_s = [M_ARGS[m][sp] for sp in M_PARAM[m]["search"]]

          for ba in product(*ba_s):
            b = bt.format(*ba)
            for sa in product(*sa_s):
              s = st.format(*sa)
              if m == "Navix":
                if da == 1:
                  path = Path(LOG_ROOT_MULTIK.format(k)) / "Navix" / d / f"output_{nrg}_{sa[0]}_navix.json"
                else:
                  path = Path(LOG_ROOT_MULTIK.format(k)) / "Navix" / d / f"{da}d" / f"output_{int(float(sel) * 100)}_{sa[0]}_navix.json"
              else:
                path = Path(LOG_ROOT_MULTIK.format(k)) / m / w / b / s
              if path.exists():
                entries.append((path, m, k, w, d, nrg, sel, b, s))
                if (len(entries) % 100 == 0):
                  print(f"Processed {len(entries)} entries")

  df = pd.DataFrame.from_records(
    entries, columns=[
      "path",
      "method",
      "k",
      "workload",
      "dataset",
      "range",
      "selectivity",
      "build",
      "search",
    ], index="path"
  )

  rec, qps, tqps, ncomp, prop, initial_ncomp = [], [], [], [], [], []
  for e in entries:
    if e[1] == "Navix":
      with open(e[0]) as f:
        stat = json.load(f)
        rec.append(stat["recall_percentage"] / 100)
        qps.append(1 / stat["avg_vector_search_time_ms"] * 1000)
        tqps.append(1 / stat["avg_execution_time_ms"] * 1000)
        ncomp.append(stat["avg_distance_computations"])
        prop.append(0)
        initial_ncomp.append(0)
      continue
    jsons = list(e[0].glob("*.json"))
    if len(jsons) == 0:
      df = df.drop(e[0])
      continue
    jsons.sort()
    with open(jsons[-1]) as f:
      try:
        stat = json.load(f)
      except:  # noqa: E722
        print(e[0])
        exit()
      # sel.append(f'{stat["aggregated"]["selectivity"]:.2f}')
      rec.append(stat["aggregated"]["recall"])
      qps.append(stat["aggregated"]["qps"])
      tqps.append(stat["aggregated"].get("tampered_qps", 0))
      ncomp.append(stat["aggregated"]["num_computations"])
      if "cluster_search_time_in_s" in stat["aggregated"] and "latency_in_s" in stat["aggregated"]:
        prop.append(stat["aggregated"]["cluster_search_time_in_s"] / stat["aggregated"]["latency_in_s"])
      else:
        prop.append(0)
      if "cluster_search_ncomp" in stat["aggregated"]:
        if "batchsz" in stat["aggregated"]:
          initial_ncomp.append(stat["aggregated"]["cluster_search_ncomp"] / stat["aggregated"]["batchsz"])
        else:
          initial_ncomp.append(stat["aggregated"]["cluster_search_ncomp"] / 100)
      elif "cg_num_computations" in stat["aggregated"]:  # For CompassPost series, add up computations together
        ncomp[-1] += stat["aggregated"]["cg_num_computations"]
        initial_ncomp.append(0)
      else:
        initial_ncomp.append(0)

    nsample, avg_qps = min(len(jsons), 3), qps[-1]
    for i in range(2, nsample + 1):
      with open(jsons[-i]) as f:
        stat = json.load(f)
        avg_qps += stat["aggregated"]["qps"]
    avg_qps /= nsample
    qps[-1] = avg_qps

  df["recall"] = rec
  df["qps"] = qps
  df["tqps"] = tqps
  df["ncomp"] = ncomp
  df["prop"] = prop
  df["initial_ncomp"] = initial_ncomp

  df.to_csv("stats-1d-multik.csv")


def draw_qps_comp_fixing_selectivity_by_k_camera(datasets, d_m_b, d_m_s, anno, prefix):
  k_s = [5, 10, 15, 20, 25, 30]
  sel_s = ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
  dataset_comp_ylim = {
    "crawl": 15000,
    "gist-dedup": 15000,
    "video-dedup": 30000,
    "glove100": 30000,
  }

  all = pd.read_csv("stats-1d-multik.csv", dtype=types)

  m_s = ["CompassPostKTh", "Navix", "SeRF"]
  for rec in [0.8, 0.85, 0.9, 0.95]:
    d_m_sel = {}
    for d in d_m_b.keys():
      d_m_sel[d] = {}
      for m in m_s:
        d_m_sel[d][m] = {}
        for sel in sel_s:
          d_m_sel[d][m][sel] = {}

    for k in k_s:
      df = all[all["k"] == k]
      for i, d in enumerate(datasets):
        data = df[df["dataset"] == d]
        for m in m_s:
          for sel in sel_s:
            for b in d_m_b[d][m]:
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
                  pos = bisect.bisect(grouped_qps["selectivity"], sel) - 1
                  if pos == -1 or grouped_qps["selectivity"][pos] != sel:
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
                pos = bisect.bisect(grouped_qps["selectivity"], sel) - 1
                if pos == -1 or grouped_qps["selectivity"][pos] != sel:
                  pos = -1
                label = f"{m}-{b}-{rec}"
                if label not in d_m_sel[d][m][sel]:  # a list by dimension
                  d_m_sel[d][m][sel][label] = {"qps": [], "ncomp": []}
                d_m_sel[d][m][sel][label]["qps"].append(grouped_qps["qps"][pos] if pos >= 0 else 0)
                d_m_sel[d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else 30000)

    for sel in sel_s:
      fig, axs = plt.subplots(2, len(datasets), layout='constrained')
      for i, d in enumerate(datasets):
        for m in m_s:
          marker = M_STYLE[m]
          for label in d_m_sel[d][m][sel].keys():
            das = k_s[:len(d_m_sel[d][m][sel][label]["qps"])]
            sc = axs[0][i].scatter(das, d_m_sel[d][m][sel][label]["qps"], label=label, **marker)
            axs[0][i].plot(das, d_m_sel[d][m][sel][label]["qps"], color=sc.get_facecolor()[0])
            if m.startswith("Compass"):
              axs[1][i].scatter(das, d_m_sel[d][m][sel][label]["total_ncomp"], label=label, **marker)
              axs[1][i].plot(das, d_m_sel[d][m][sel][label]["total_ncomp"], color=sc.get_facecolor()[0])
            else:
              axs[1][i].scatter(das, d_m_sel[d][m][sel][label]["ncomp"], label=label, **marker)
              axs[1][i].plot(das, d_m_sel[d][m][sel][label]["ncomp"], color=sc.get_facecolor()[0])

        dt = d.split("-")[0].upper()
        # axs[0][i].set_xlabel('$k$')
        axs[0][i].set_xticks(k_s)
        if i == 0:
          axs[0][i].set_ylabel('QPS')
        axs[0][i].set_title(f"{dt}, Recall-{rec:.3g}")
        axs[1][i].set_xlabel('$k$')
        axs[1][i].set_xticks(k_s)
        if i == 0:
          axs[1][i].set_ylabel('# Comp')
        auto_bottom, auto_top = axs[1][i].get_ylim()
        axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
        # axs[1][i].set_title(f"{dt}, Recall-{rec:.3g}")

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
      path = Path(f"{prefix}/Sel-{float(sel):.3g}-Recall-{rec:.3g}-{anno}-All-QPS-Comp.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      # plt.grid(True)
      fig.savefig(path, dpi=200)
      plt.close("all")


def draw_qps_comp_wrt_recall_by_selectivity_camera_shrinked(da, datasets, methods, anno, *, d_m_b={}, d_m_s={}, prefix="figures", ranges=[]):
  xlim, xticks = [0.8, 1], [0.8, 0.9, 1]
  if ranges:  # ablations study
    xlim, xticks = [0.4, 1], [0.4, 0.6, 0.8, 1]
  df = pd.read_csv(f"stats-{da}d.csv", dtype=types)
  df = df.fillna('')
  dataset_comp_ylim = {
    "crawl": 10000,
    "gist-dedup": 10000,
    "video-dedup": 30000,
    "glove100": 30000,
  }

  selected_efs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 500, 600, 800, 1000]
  selected_efs = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 500, 600, 800, 1000]

  rg_d_b = {}
  for rg in ranges if ranges else DA_RANGE[da]:
    rg_d_b[rg] = {}
    fig, axs = plt.subplots(2, len(datasets), layout='tight')
    fig.set_size_inches(8, 4)
    for ax in axs.flat:
      ax.set_box_aspect(1)
      ax.tick_params(axis='y', rotation=45)
    for i, d in enumerate(datasets):
      rg_d_b[rg][d] = {}
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
            rg_d_b[rg][d][b] = {}
            for nrel in d_m_s.get(d, {}).get(m, {}).get("nrel", compass_args["nrel"]):
              _selected_efs = D_ARGS[d]["efs"]
              selected_search = [f"efs_{efs}_nrel_{nrel}_batch_k_20_initial_efs_20_delta_efs_20" for efs in _selected_efs]
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].isin(selected_search)]
              rec_qps_comp = data_by_m_b_nrel[["recall", "qps", "ncomp", "initial_ncomp", "search"]].sort_values(["recall"], ascending=[True])
              rec_qps_comp["total_ncomp"] = rec_qps_comp["initial_ncomp"] + rec_qps_comp["ncomp"]

              recall_above = rec_qps_comp[rec_qps_comp["recall"].gt(xlim[0])]
              # Need comment out following two lines when plotting for ablation due to the index-out-of-range.
              rg_d_b[rg][d][b]["recall"] = rec_qps_comp[rec_qps_comp["recall"].gt(0.9)]["recall"].to_list()[0]
              rg_d_b[rg][d][b]["search"] = rec_qps_comp[rec_qps_comp["recall"].gt(0.9)]["search"].to_list()[0]
              axs[0][i].plot(recall_above["recall"], recall_above["qps"], **marker)
              axs[0][i].scatter(recall_above["recall"], recall_above["qps"], label=f"{m}-{b}-nrel_{nrel}", **marker)
              axs[1][i].plot(recall_above["recall"], recall_above["total_ncomp"], **marker)
              axs[1][i].scatter(recall_above["recall"], recall_above["total_ncomp"], label=f"{m}-{b}-nrel_{nrel}", **marker)
          else:
            selected_search = [f"efs_{efs}" for efs in selected_efs]
            data_by_m_b = data_by_m_b[data_by_m_b["search"].isin(selected_search)]
            recall_qps = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
            recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
            axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
            axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", **marker)
            if m in {"Milvus", "Weaviate"}: continue #  # noqa: E701
            recall_ncomp = data_by_m_b[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
            recall_ncomp = recall_ncomp[recall_ncomp["recall"].gt(xlim[0])].to_numpy()
            axs[1][i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1], **marker)
            axs[1][i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1], label=f"{m}-{b}", **marker)

          dt = d.split("-")[0].upper()
          # axs[0][i].set_xlabel('Recall')
          axs[0][i].set_xticks(xticks)
          axs[0][i].set_xticklabels([])
          if i == 0:
            axs[0][i].set_ylabel('QPS')
          axs[0][i].set_title("{}, {:.1%}".format(dt, sel))
          axs[1][i].set_xlabel('Recall')
          axs[1][i].set_xticks(xticks)
          if i == 0:
            axs[1][i].set_ylabel('# Comp')
          auto_bottom, auto_top = axs[1][i].get_ylim()
          axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
          # axs[1][i].set_title("{}, Passrate-{:.1%}".format(dt, sel))

      bottom = 0.05
      plt.tight_layout(rect=[0, bottom, 1, 1], w_pad=0.05, h_pad=0.05)
      unique_labels = {}
      for ax in axs[0]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
          label = label.split("-")[0]
          if label.startswith("Compass"):
            label = label if label != "CompassPostKTh" else "Compass"
          if label.startswith("SeRF"):
            label = "SeRF"
          if label.startswith("Navix"):
            label = "NaviX"
          if label not in unique_labels:
            unique_labels[label] = handle
        # ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper right")
      for ax in axs[1]:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
          label = label.split("-")[0]
          if label.startswith("Compass"):
            label = label if label != "CompassPostKTh" else "Compass"
          if label.startswith("SeRF"):
            label = "SeRF"
          if label.startswith("Navix"):
            label = "NaviX"
          if label not in unique_labels:
            unique_labels[label] = handle
        # ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper left")
      # Put a legend below current axis
      fig.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc='outside lower center',
        bbox_to_anchor=(0.5, 0),
        fancybox=True,
        ncol=len(unique_labels),
      )
      # plt.grid(True)
      path = Path(f"{prefix}/Shrinked-All-{anno}-{rg}-QPS-Comp-Recall.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      fig.savefig(path, dpi=200)
      plt.close("all")

  with open(f"{prefix}/All-{anno}-QPS-Comp-Recall-Search.json", "w") as f:
    json.dump(rg_d_b, f, indent=4)


def draw_qps_comp_fixing_dimension_selectivity_by_dimension_camera_shrinked(datasets, d_m_b, d_m_s, anno, prefix):
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

  rec_d_m_sel = {}
  for rec in [0.8, 0.85, 0.9, 0.95]:
    rec_d_m_sel[rec] = {}
    for d in d_m_b.keys():
      rec_d_m_sel[rec][d] = {}
      for m in d_m_b[d].keys():
        rec_d_m_sel[rec][d][m] = {}
        for sel in sel_s:
          rec_d_m_sel[rec][d][m][sel] = {}

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
                  if label not in rec_d_m_sel[rec][d][m][sel]:  # a list by dimension
                    rec_d_m_sel[rec][d][m][sel][label] = {"qps": [], "ncomp": [], "total_ncomp": []}
                  rec_d_m_sel[rec][d][m][sel][label]["qps"].append(grouped_qps["qps"][pos] if pos >= 0 else 0)
                  rec_d_m_sel[rec][d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else 30000)
                  rec_d_m_sel[rec][d][m][sel][label]["total_ncomp"].append(grouped_total_comp["total_ncomp"][pos] if pos >= 0 else 30000)
              else:
                rec_sel_qps_comp = data_by_m_b[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
                grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
                grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
                pos = bisect.bisect(grouped_qps["selectivity"], interval[sel][da - 1]) - 1
                if pos == -1 or grouped_qps["selectivity"][pos] != interval[sel][da - 1]:
                  pos = -1
                  fallout_qps = rec_sel_qps_comp[rec_sel_qps_comp["selectivity"] == interval[sel][da - 1]]["qps"].min()
                  fallout_ncomp = rec_sel_qps_comp[rec_sel_qps_comp["selectivity"] == interval[sel][da - 1]]["ncomp"].max()
                label = f"{m}-{b}-{rec}"
                if label not in rec_d_m_sel[rec][d][m][sel]:
                  rec_d_m_sel[rec][d][m][sel][label] = {"qps": [], "ncomp": [], "fallout": []}
                rec_d_m_sel[rec][d][m][sel][label]["qps"].append(grouped_qps["qps"][pos] if pos >= 0 else fallout_qps)
                rec_d_m_sel[rec][d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else fallout_ncomp)
                if pos == -1:
                  rec_d_m_sel[rec][d][m][sel][label]["fallout"].append(len(rec_d_m_sel[rec][d][m][sel][label]["qps"]))

  with open(f"{prefix}/conjunction_rec_d_m_sel.json", "w") as f:
    json.dump(rec_d_m_sel, f, indent=4)

  for rec in [0.8, 0.85, 0.9, 0.95]:
    for sel in sel_s:
      fig, axs = plt.subplots(2, len(datasets), layout='tight')
      fig.set_size_inches(8, 4)
      for ax in axs.flat:
        ax.set_box_aspect(1)
        ax.tick_params(axis='y', rotation=45)
      for i, d in enumerate(datasets):
        for m in d_m_b[d].keys():
          marker = M_STYLE[m]
          for label in rec_d_m_sel[rec][d][m][sel].keys():
            das = DA_S[:len(rec_d_m_sel[rec][d][m][sel][label]["qps"])]
            sc = axs[0][i].scatter(das, rec_d_m_sel[rec][d][m][sel][label]["qps"], label=label, **marker)
            axs[0][i].plot(das, rec_d_m_sel[rec][d][m][sel][label]["qps"], color=sc.get_facecolor()[0])
            if m.startswith("Compass"):
              axs[1][i].scatter(das, rec_d_m_sel[rec][d][m][sel][label]["total_ncomp"], label=label, **marker)
              axs[1][i].plot(das, rec_d_m_sel[rec][d][m][sel][label]["total_ncomp"], color=sc.get_facecolor()[0])
            elif m not in {"Milvus", "Weaviate"}:
              axs[1][i].scatter(das, rec_d_m_sel[rec][d][m][sel][label]["ncomp"], label=label, **marker)
              axs[1][i].plot(das, rec_d_m_sel[rec][d][m][sel][label]["ncomp"], color=sc.get_facecolor()[0])
            for tick in rec_d_m_sel[rec][d][m][sel][label].get("fallout", []):
              # mark a cross sign at (tick, fallout_qps)
              axs[0][i].scatter(tick, rec_d_m_sel[rec][d][m][sel][label]["qps"][tick - 1], s=100, color=sc.get_facecolor()[0], marker='x')
              if m in {"Milvus", "Weaviate"}: continue  # noqa: E701
              axs[1][i].scatter(tick, rec_d_m_sel[rec][d][m][sel][label]["ncomp"][tick - 1], s=100, color=sc.get_facecolor()[0], marker='x')
        dt = d.split("-")[0].upper()
        # axs[0][i].set_xlabel('Dimension')
        axs[0][i].set_xticks(DA_S)
        axs[0][i].set_xticklabels([])
        if i == 0:
          axs[0][i].set_ylabel('QPS')
        axs[0][i].set_title(f"{dt}, Rec.-{rec:.3g}")
        axs[1][i].set_xlabel('Dimension')
        if i == 0:
          axs[1][i].set_ylabel('# Comp')
        auto_bottom, auto_top = axs[1][i].get_ylim()
        axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
        axs[1][i].set_xticks(DA_S)
        # axs[1][i].set_title(f"{dt}, Recall-{rec:.3g}")

      bottom = 0.05
      plt.tight_layout(rect=[0, bottom, 1, 1], w_pad=0.05, h_pad=0.05)
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
        # ax.legend(unique_labels.values(), unique_labels.keys(), loc="best")
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
        # ax.legend(unique_labels.values(), unique_labels.keys(), loc="best")
      # fig.legend(unique_labels.values(), unique_labels.keys(), loc="upper right")
      # Put a legend below current axis
      fig.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc='outside lower center',
        bbox_to_anchor=(0.5, 0),
        fancybox=True,
        ncol=len(unique_labels),
      )
      path = Path(f"{prefix}/Shrinked-Sel-{sel:.3g}-Recall-{rec:.3g}-{anno}-All-QPS-Comp.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      # plt.grid(True)
      fig.savefig(path, dpi=200)
      plt.close("all")

  for rec_s, dataset_s in zip([(0.85, 0.95)], [("video-dedup", "gist-dedup")]):
    for sel in sel_s:
      fig, axs = plt.subplots(2, len(datasets), layout='tight')
      fig.set_size_inches(8, 4)
      for ax in axs.flat:
        ax.set_box_aspect(1)
        ax.tick_params(axis='y', rotation=45)
      for rec_i, rec in enumerate(rec_s):
        for d_i, d in enumerate(dataset_s):
          i = rec_i * len(dataset_s) + d_i
          for m in d_m_b[d].keys():
            marker = M_STYLE[m]
            for label in rec_d_m_sel[rec][d][m][sel].keys():
              das = DA_S[:len(rec_d_m_sel[rec][d][m][sel][label]["qps"])]
              sc = axs[0][i].scatter(das, rec_d_m_sel[rec][d][m][sel][label]["qps"], label=label, **marker)
              axs[0][i].plot(das, rec_d_m_sel[rec][d][m][sel][label]["qps"], color=sc.get_facecolor()[0])
              if m.startswith("Compass"):
                axs[1][i].scatter(das, rec_d_m_sel[rec][d][m][sel][label]["total_ncomp"], label=label, **marker)
                axs[1][i].plot(das, rec_d_m_sel[rec][d][m][sel][label]["total_ncomp"], color=sc.get_facecolor()[0])
              elif m not in {"Milvus", "Weaviate"}:
                axs[1][i].scatter(das, rec_d_m_sel[rec][d][m][sel][label]["ncomp"], label=label, **marker)
                axs[1][i].plot(das, rec_d_m_sel[rec][d][m][sel][label]["ncomp"], color=sc.get_facecolor()[0])
              for tick in rec_d_m_sel[rec][d][m][sel][label].get("fallout", []):
                # mark a cross sign at (tick, fallout_qps)
                axs[0][i].scatter(tick, rec_d_m_sel[rec][d][m][sel][label]["qps"][tick - 1], s=100, color=sc.get_facecolor()[0], marker='x')
                if m in {"Milvus", "Weaviate"}: continue  # noqa: E701
                axs[1][i].scatter(tick, rec_d_m_sel[rec][d][m][sel][label]["ncomp"][tick - 1], s=100, color=sc.get_facecolor()[0], marker='x')
          dt = d.split("-")[0].upper()
          # axs[0][i].set_xlabel('Dimension')
          axs[0][i].set_xticks(DA_S)
          axs[0][i].set_xticklabels([])
          if i == 0:
            axs[0][i].set_ylabel('QPS')
          axs[0][i].set_title(f"{dt}, Rec.-{rec:.3g}")
          axs[1][i].set_xlabel('Dimension')
          if i == 0:
            axs[1][i].set_ylabel('# Comp')
          auto_bottom, auto_top = axs[1][i].get_ylim()
          axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
          axs[1][i].set_xticks(DA_S)
          # axs[1][i].set_title(f"{dt}, Recall-{rec:.3g}")

      bottom = 0.05
      plt.tight_layout(rect=[0, bottom, 1, 1], w_pad=0.05, h_pad=0.05)
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
        # ax.legend(unique_labels.values(), unique_labels.keys(), loc="best")
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
        # ax.legend(unique_labels.values(), unique_labels.keys(), loc="best")
      # fig.legend(unique_labels.values(), unique_labels.keys(), loc="upper right")
      fig.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc='outside lower center',
        bbox_to_anchor=(0.5, 0),
        fancybox=True,
        ncol=len(unique_labels),
      )
      path = Path(f"{prefix}/Shrinked-Sel-{sel:.3g}-Recall-{rec_s[0]:.3g}-{rec_s[1]:.3g}-{anno}-All-QPS-Comp.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      # plt.grid(True)
      fig.savefig(path, dpi=200)
      plt.close("all")


def draw_qps_comp_with_disjunction_by_dimension_camera_shrinked(datasets, d_m_b, d_m_s, anno, prefix):
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

  rec_d_m = {}
  for rec in [0.8, 0.85, 0.9, 0.95]:
    rec_d_m[rec] = {}
    for d in d_m_b.keys():
      rec_d_m[rec][d] = {}
      for m in d_m_b[d].keys():
        rec_d_m[rec][d][m] = {}
        for sel in sel_s:
          rec_d_m[rec][d][m][sel] = {}

    for ndis in ndisjunctions:
      dataset_comp_ylim_4d = {
        "crawl": 10000,
        "gist-dedup": 10000,
        "video-dedup": 80000,
        "glove100": 80000,
      }
      if ndis == 4: dataset_comp_ylim = dataset_comp_ylim_4d  # noqa: E701
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
                  data_by_m_b = data[(data["method"] == m + "+OR4") & (data["build"] == b)]
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
                  if label not in rec_d_m[rec][d][m][sel]:  # a list by dimension
                    rec_d_m[rec][d][m][sel][label] = {"qps": [], "ncomp": [], "total_ncomp": []}
                  rec_d_m[rec][d][m][sel][label]["qps"].append(grouped_qps["qps"][pos] if pos >= 0 else 0)
                  rec_d_m[rec][d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else 30000)
                  rec_d_m[rec][d][m][sel][label]["total_ncomp"].append(grouped_total_comp["total_ncomp"][pos] if pos >= 0 else 30000)
              else:
                rec_sel_qps_comp = data_by_m_b[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
                grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
                grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
                pos = bisect.bisect(grouped_qps["selectivity"], interval[sel][ndis - 1]) - 1
                if pos == -1 or grouped_qps["selectivity"][pos] != interval[sel][ndis - 1]:
                  pos = -1
                  fallout_qps = rec_sel_qps_comp[rec_sel_qps_comp["selectivity"] == interval[sel][ndis - 1]]["qps"].min()
                  fallout_ncomp = rec_sel_qps_comp[rec_sel_qps_comp["selectivity"] == interval[sel][ndis - 1]]["ncomp"].max()
                label = f"{m}-{b}-{rec}"
                if label not in rec_d_m[rec][d][m][sel]:
                  rec_d_m[rec][d][m][sel][label] = {"qps": [], "ncomp": [], "fallout": []}
                rec_d_m[rec][d][m][sel][label]["qps"].append(grouped_qps["qps"][pos] if pos >= 0 else fallout_qps)
                rec_d_m[rec][d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else fallout_ncomp)
                if pos == -1:
                  rec_d_m[rec][d][m][sel][label]["fallout"].append(len(rec_d_m[rec][d][m][sel][label]["qps"]))

  with open(f"{prefix}/disjunction_rec_d_m_sel.json", "w") as f:
    json.dump(rec_d_m, f, indent=4)

  for rec in [0.8, 0.85, 0.9, 0.95]:
    for sel in sel_s:
      fig, axs = plt.subplots(2, len(datasets), layout='tight')
      fig.set_size_inches(8, 4)
      for ax in axs.flat:
        ax.set_box_aspect(1)
        ax.tick_params(axis='y', rotation=45)
      for i, d in enumerate(datasets):
        for m in d_m_b[d].keys():
          marker = M_STYLE[m]
          for label in rec_d_m[rec][d][m][sel].keys():
            sc = axs[0][i].scatter(ndisjunctions, rec_d_m[rec][d][m][sel][label]["qps"], label=label, **marker)
            axs[0][i].plot(ndisjunctions, rec_d_m[rec][d][m][sel][label]["qps"], color=sc.get_facecolor()[0])
            if m.startswith("Compass"):
              axs[1][i].scatter(ndisjunctions, rec_d_m[rec][d][m][sel][label]["total_ncomp"], label=label, **marker)
              axs[1][i].plot(ndisjunctions, rec_d_m[rec][d][m][sel][label]["total_ncomp"], color=sc.get_facecolor()[0])
            elif m not in {"Milvus", "Weaviate"}:
              axs[1][i].scatter(ndisjunctions, rec_d_m[rec][d][m][sel][label]["ncomp"], label=label, **marker)
              axs[1][i].plot(ndisjunctions, rec_d_m[rec][d][m][sel][label]["ncomp"], color=sc.get_facecolor()[0])
            for tick in rec_d_m[rec][d][m][sel][label].get("fallout", []):
              # mark a cross sign at (tick, fallout_qps)
              axs[0][i].scatter(tick, rec_d_m[rec][d][m][sel][label]["qps"][tick - 1], s=100, color=sc.get_facecolor()[0], marker='x')
              if m in {"Milvus", "Weaviate"}: continue  # noqa: E701
              axs[1][i].scatter(tick, rec_d_m[rec][d][m][sel][label]["ncomp"][tick - 1], s=100, color=sc.get_facecolor()[0], marker='x')

        dt = d.split("-")[0].upper()
        # axs[0][i].set_xlabel('Dimension')
        axs[0][i].set_xticks(ndisjunctions)
        axs[0][i].set_xticklabels([])
        if i == 0:
          axs[0][i].set_ylabel('QPS')
        axs[0][i].set_title(f"{dt}, Rec.-{rec:.3g}")
        axs[1][i].set_xlabel('Dimension')
        if i == 0:
          axs[1][i].set_ylabel('# Comp')
        auto_bottom, auto_top = axs[1][i].get_ylim()
        axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
        axs[1][i].set_xticks(ndisjunctions)
        # axs[1][i].set_title(f"{dt}, Recall-{rec:.3g}")

      bottom = 0.05
      plt.tight_layout(rect=[0, bottom, 1, 1], w_pad=0.05, h_pad=0.05)
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
        # ax.legend(unique_labels.values(), unique_labels.keys(), loc="best")
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
        # ax.legend(unique_labels.values(), unique_labels.keys(), loc="best")
      # fig.legend(unique_labels.values(), unique_labels.keys(), loc="upper right")
      # Put a legend below current axis
      fig.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc='outside lower center',
        bbox_to_anchor=(0.5, 0),
        fancybox=True,
        ncol=len(unique_labels),
      )
      path = Path(f"{prefix}/Shrinked-Sel-{sel:.3g}-Recall-{rec:.3g}-{anno}-All-QPS-Comp.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      # plt.grid(True)
      fig.savefig(path, dpi=200)
      plt.close("all")

  for rec_s, dataset_s in zip([(0.85, 0.95)], [("video-dedup", "gist-dedup")]):
    for sel in sel_s:
      fig, axs = plt.subplots(2, len(datasets), layout='tight')
      fig.set_size_inches(8, 4)
      for ax in axs.flat:
        ax.set_box_aspect(1)
        ax.tick_params(axis='y', rotation=45)
      for rec_i, rec in enumerate(rec_s):
        for d_i, d in enumerate(dataset_s):
          i = rec_i * len(dataset_s) + d_i
          for m in d_m_b[d].keys():
            marker = M_STYLE[m]
            for label in rec_d_m[rec][d][m][sel].keys():
              sc = axs[0][i].scatter(ndisjunctions, rec_d_m[rec][d][m][sel][label]["qps"], label=label, **marker)
              axs[0][i].plot(ndisjunctions, rec_d_m[rec][d][m][sel][label]["qps"], color=sc.get_facecolor()[0])
              if m.startswith("Compass"):
                axs[1][i].scatter(ndisjunctions, rec_d_m[rec][d][m][sel][label]["total_ncomp"], label=label, **marker)
                axs[1][i].plot(ndisjunctions, rec_d_m[rec][d][m][sel][label]["total_ncomp"], color=sc.get_facecolor()[0])
              elif m not in {"Milvus", "Weaviate"}:
                axs[1][i].scatter(ndisjunctions, rec_d_m[rec][d][m][sel][label]["ncomp"], label=label, **marker)
                axs[1][i].plot(ndisjunctions, rec_d_m[rec][d][m][sel][label]["ncomp"], color=sc.get_facecolor()[0])
              for tick in rec_d_m[rec][d][m][sel][label].get("fallout", []):
                # mark a cross sign at (tick, fallout_qps)
                axs[0][i].scatter(tick, rec_d_m[rec][d][m][sel][label]["qps"][tick - 1], s=100, color=sc.get_facecolor()[0], marker='x')
                if m in {"Milvus", "Weaviate"}: continue  # noqa: E701
                axs[1][i].scatter(tick, rec_d_m[rec][d][m][sel][label]["ncomp"][tick - 1], s=100, color=sc.get_facecolor()[0], marker='x')

          dt = d.split("-")[0].upper()
          # axs[0][i].set_xlabel('Dimension')
          axs[0][i].set_xticks(ndisjunctions)
          axs[0][i].set_xticklabels([])
          if i == 0:
            axs[0][i].set_ylabel('QPS')
          axs[0][i].set_title(f"{dt}, Rec.-{rec:.3g}")
          axs[1][i].set_xlabel('Dimension')
          if i == 0:
            axs[1][i].set_ylabel('# Comp')
          auto_bottom, auto_top = axs[1][i].get_ylim()
          axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
          axs[1][i].set_xticks(ndisjunctions)
          # axs[1][i].set_title(f"{dt}, Recall-{rec:.3g}")

      bottom = 0.05
      plt.tight_layout(rect=[0, bottom, 1, 1], w_pad=0.05, h_pad=0.05)
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
        # ax.legend(unique_labels.values(), unique_labels.keys(), loc="best")
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
        # ax.legend(unique_labels.values(), unique_labels.keys(), loc="best")
      # fig.legend(unique_labels.values(), unique_labels.keys(), loc="upper right")
      # Put a legend below current axis
      fig.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc='outside lower center',
        bbox_to_anchor=(0.5, 0),
        fancybox=True,
        ncol=len(unique_labels),
      )
      path = Path(f"{prefix}/Shrinked-Sel-{sel:.3g}-Recall-{rec_s[0]:.3g}-{rec_s[1]:.3g}-{anno}-All-QPS-Comp.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      # plt.grid(True)
      fig.savefig(path, dpi=200)
      plt.close("all")


def draw_time_breakdown():
  ranges = [1, 30, 80]
  datasets = ["CRAWL", "VIDEO", "GIST", "GLOVE100"]
  total_time = {
    1: np.array([2.78, 2.81, 4.36, 7.87]),
    30: np.array([1.82, 8.46, 7.49, 10.34]),
    80: np.array([1.24, 7.69, 6.6, 9.66]),
  }
  graph_time = {
    1: np.array([0.1 + 0.07, 0.04 + 0.15, 0, 0.09]),
    30: np.array([1.24 + 0.17, 6.33 + 0.99, 5.68 + 0.88, 8.17 + 0.16]),
    80: np.array([0.61 + 0.44, 6.33 + 0.97, 5.29 + 0.9, 7.97 + 0.65]),
  }
  clusb_time = {
    1: np.array([
      1.32 + 0.1,
      1.12 + 0.45,
      2.35 + 0.27,
      5.59 + 0.13,
    ]),
    30: np.array([
      0.24 + 0,
      0.46 + 0,
      0.32,
      0.3,
    ]),
    80: np.array([
      0.05 + 0,
      0.08,
      0,
      0.30,
    ]),
  }
  graph_comp_time = {
    1: np.array([
      0.07 + 0,
      0.15 + 0,
      0,
      0.04 + 0.09
    ]),
    30: np.array([
      0.56 + 0.07,
      4.46 + 0.42,
      3.92 + 0.44,
      2.21 + 0.13,
    ]),
    80: np.array([
      0.39 + 0.24,
      4.66 + 0.54,
      3.73 + 0.57,
      2.97 + 0.22
    ]),
  }
  cg_comp_time = {
    1: np.array([
      0.41 + 0.02,
      0.56 + 0.19,
      1 + 0.12,
      0.85+0.04
    ]),
    30: np.array([
      0.05,
      0.15,
      0.16,
      0.09,
    ]),
    80: np.array([
      0.05,
      0.04,
      0,
      0.09
    ]),
  }
  ivf_comp_time = {
    1: np.array([
      0.38,
      0.45,
      0.89,
      0.39,
    ]),
    30: np.array([
      0.02,
      0.38,
      0.36,
      0.34
    ]),
    80: np.array([
      0,
      0.08,
      0.04,
      0.04
    ]),
  }
  filter_timer = {
    1: np.array([
      0,
      0,
      0,
      0,
    ]),
    30: np.array([
      0,
      0,
      0,
      0,
    ]),
    80: np.array([
      0,
      0,
      0,
      0,
    ]),
  }

  fig, axs = plt.subplots(1, len(ranges))
  axs = axs.flatten()
  _bottom = 0.05
  plt.tight_layout(rect=[0, _bottom, 1, 1], w_pad=0.05, h_pad=0.05)
  fig.set_size_inches(10, 4)
  for rg, ax in zip(ranges, axs):

    cg_comp_time[rg] += ivf_comp_time[rg]
    ax.set_box_aspect(1)

    bottom = np.zeros(4)
    ax.bar(datasets, np.ones(4), label='Others', bottom=bottom)
    ax.bar(datasets, graph_time[rg] / total_time[rg], label='Proximity Graph', bottom=bottom)
    # ax.bar(datasets, graph_comp_time[rg] / total_time[rg], label='Graph Comp', bottom=bottom)

    bottom += graph_time[rg] / total_time[rg]
    ax.bar(datasets, clusb_time[rg] / total_time[rg], label='Clustered B+-trees', bottom=bottom)
    # ax.bar(datasets, cg_comp_time[rg] / total_time[rg], label='Cluster Comp', bottom=bottom)

    ax.set_title(f'Passrate {rg}%, Recall 0.9')

  unique_labels = {}
  for ax in axs:
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
      # label = label.split("-")[0]
      # if label.startswith("Compass"):
      #   label = "Compass"
      # if label.startswith("SeRF"):
      #   label = "SeRF"
      unique_labels[label] = handle
  fig.legend(
    unique_labels.values(),
    unique_labels.keys(),
    loc='outside lower center',
    bbox_to_anchor=(0.5, 0),
    fancybox=True,
    ncol=3,
  )
  fig.savefig('time_breakdown.jpg', dpi=200)
