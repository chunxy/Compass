import bisect
import json
from functools import reduce
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import (
  COMPASS_METHODS,
  DA_RANGE,
  DA_S,
  DA_SEL,
  DATASETS,
  M_ARGS,
  M_DA_RUN,
  M_STYLE,
  M_PARAM,
  M_WORKLOAD,
  METHODS,
  compass_args,
  D_ARGS,
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

xlim = [0.6, 1]


def summarize():
  for da in DA_S:
    entries = []
    for m in METHODS:
      if da not in M_DA_RUN[m]: continue  # noqa: E701
      for d in DATASETS:
        for itvl in M_DA_RUN[m][da]:
          if m in COMPASS_METHODS:
            w = M_WORKLOAD[m].format(d, *map(lambda ele: "-".join(map(str, ele)), itvl))
          else:
            w = M_WORKLOAD[m].format(d, "-".join(map(str, itvl)))
          bt = "_".join([f"{bp}_{{}}" for bp in M_PARAM[m]["build"]])
          st = "_".join([f"{sp}_{{}}" for sp in M_PARAM[m]["search"]])
          for ba in product(*[D_ARGS[d].get(bp, M_ARGS[m][bp]) for bp in M_PARAM[m]["build"]]):
            b = bt.format(*ba)
            for sa in product(*[D_ARGS[d].get(sp, M_ARGS[m][sp]) for sp in M_PARAM[m]["search"]]):
              s = st.format(*sa)
              if m in COMPASS_METHODS:
                nrg = "-".join([f"{(r - l) // 100}" for l, r in zip(*itvl)])  # noqa: E741
                sel = f"{reduce(lambda a, b: a * b, [(r - l) / 10000 for l, r in zip(*itvl)], 1.):.3g}"  # noqa: E741
              else:
                nrg = "-".join(map(str, itvl))
                sel = f"{reduce(lambda a, b: a * b, map(lambda x: x / 100, itvl), 1.):.3g}"
              path = LOG_ROOT / m / w / b / s
              if path.exists():
                entries.append((path, m, w, d, nrg, sel, b, s))
    df = pd.DataFrame.from_records(
      entries, columns=[
        "path",
        "method",
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
      jsons = list(e[0].glob("*.json"))
      if len(jsons) == 0:
        df = df.drop(e[0])
        continue
      jsons.sort()
      with open(jsons[-1]) as f:
        stat = json.load(f)
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
        else:
          initial_ncomp.append(0)

    df["recall"] = rec
    df["qps"] = qps
    df["tqps"] = tqps
    df["ncomp"] = ncomp
    df["prop"] = prop
    df["initial_ncomp"] = initial_ncomp

    df.to_csv(f"stats-{da}d.csv")


def draw_qps_comp_wrt_recall_by_dataset_selectivity(da, datasets, methods, anno, *, d_m_b={}, d_m_s={}, prefix="figures"):
  df = pd.read_csv(f"stats-{da}d.csv", dtype=types)

  for d in datasets:
    for rg in DA_RANGE[da]:
      selector = ((df["dataset"] == d) & (df["range"] == rg))
      if not selector.any():
        continue

      data = df[selector]
      sel = float(data["selectivity"].unique()[0])
      fig, axs = plt.subplots(1, 4, layout='constrained')
      for m in d_m_b[d].keys() if d in d_m_b else methods:
        marker = M_STYLE[m]
        for b in d_m_b.get(d, {}).get(m, data[data["method"] == m].build.unique()):
          data_by_m_b = data[(data["method"] == m) & (data["build"].str.contains(b))]
          if m.startswith("Compass"):
            for nrel in d_m_s.get(d, {}).get(m, {}).get("nrel", compass_args["nrel"]):
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].str.contains(f"nrel_{nrel}")]
              recall_qps = data_by_m_b_nrel[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
              recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
              axs[0].plot(recall_qps[:, 0], recall_qps[:, 1])
              axs[0].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)

              recall_comp = data_by_m_b_nrel[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
              recall_comp = recall_comp[recall_comp["recall"].gt(xlim[0])].to_numpy()
              axs[1].plot(recall_comp[:, 0], recall_comp[:, 1])
              axs[1].scatter(recall_comp[:, 0], recall_comp[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)
              recall_i_comp = data_by_m_b_nrel[["recall", "initial_ncomp"]].sort_values(["recall", "initial_ncomp"], ascending=[True, True])
              recall_i_comp = recall_i_comp[recall_i_comp["recall"].gt(xlim[0])].to_numpy()
              axs[1].plot(recall_i_comp[:, 0], recall_i_comp[:, 1] + recall_comp[:, 1], linestyle="--")
              axs[1].scatter(recall_i_comp[:, 0], recall_i_comp[:, 1] + recall_comp[:, 1], label=f"total-{m}-{b}-nrel_{nrel}", **marker)

              recall_tqps = data_by_m_b_nrel[["recall", "tqps"]].sort_values(["recall", "tqps"], ascending=[True, False])
              recall_tqps = recall_tqps[recall_tqps["recall"].gt(xlim[0])].to_numpy()
              axs[2].plot(recall_tqps[:, 0], recall_tqps[:, 1])
              axs[2].scatter(recall_tqps[:, 0], recall_tqps[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)

              recall_prop = data_by_m_b_nrel[["recall", "prop"]].sort_values(["recall", "prop"], ascending=[True, True])
              recall_prop = recall_prop[recall_prop["recall"].gt(xlim[0])].to_numpy()
              axs[3].plot(recall_prop[:, 0], recall_prop[:, 1])
              axs[3].scatter(recall_prop[:, 0], recall_prop[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)

          else:
            recall_qps = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
            recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
            axs[0].plot(recall_qps[:, 0], recall_qps[:, 1])
            axs[0].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", **marker)
            axs[2].plot(recall_qps[:, 0], recall_qps[:, 1])
            axs[2].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", **marker)

            recall_comp = data_by_m_b[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
            recall_comp = recall_comp[recall_comp["recall"].gt(xlim[0])].to_numpy()
            axs[1].plot(recall_comp[:, 0], recall_comp[:, 1])
            axs[1].scatter(recall_comp[:, 0], recall_comp[:, 1], label=f"{m}-{b}", **marker)

          axs[0].set_xlabel('Recall')
          axs[0].set_ylabel('QPS')
          axs[0].set_title("{}, Selectivity-{:.1%}".format(d.capitalize(), sel))
          axs[1].set_xlabel('Recall')
          axs[1].set_ylabel('# Comp')
          axs[1].set_title("{}, Selectivity-{:.1%}".format(d.capitalize(), sel))
          axs[2].set_xlabel('Recall')
          axs[2].set_ylabel('Tampered QPS')
          axs[2].set_title("{}, Selectivity-{:.1%}".format(d.capitalize(), sel))
          axs[3].set_xlabel('Recall')
          axs[3].set_ylabel('Initial / Racing')
          axs[3].set_title("{}, Selectivity-{:.1%}".format(d.capitalize(), sel))

      fig.set_size_inches(15, 3)
      unique_labels = {}
      for ax in axs.flat:
        ax.set_xlim(xlim)
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
          if label not in unique_labels:
            unique_labels[label] = handle
      fig.legend(unique_labels.values(), unique_labels.keys(), loc="outside right upper")
      path = Path(f"{prefix}/{d.upper()}/{d.upper()}-{anno}-{rg}-QPS-Comp-Recall.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      fig.savefig(path, dpi=200)
      plt.close()


def draw_qps_comp_wrt_recall_by_selectivity(da, datasets, methods, anno, *, d_m_b={}, d_m_s={}, prefix="figures"):
  df = pd.read_csv(f"stats-{da}d.csv", dtype=types)

  for rg in DA_RANGE[da]:
    fig, axs = plt.subplots(4, len(datasets), layout='constrained')
    for i, d in enumerate(datasets):
      selector = ((df["dataset"] == d) & (df["range"] == rg))
      if not selector.any():
        continue

      data = df[selector]
      sel = float(data["selectivity"].unique()[0])
      for m in d_m_b[d].keys() if d in d_m_b else methods:
        marker = M_STYLE[m]
        for b in d_m_b.get(d, {}).get(m, data[data["method"] == m].build.unique()):
          data_by_m_b = data[(data["method"] == m) & (data["build"].str.contains(b))]
          if m.startswith("Compass"):
            for nrel in d_m_s.get(d, {}).get(m, {}).get("nrel", compass_args["nrel"]):
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].str.contains(f"nrel_{nrel}")]
              recall_qps = data_by_m_b_nrel[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
              recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
              axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1])
              axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)

              recall_comp = data_by_m_b_nrel[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
              recall_comp = recall_comp[recall_comp["recall"].gt(xlim[0])].to_numpy()
              axs[1][i].plot(recall_comp[:, 0], recall_comp[:, 1])
              axs[1][i].scatter(recall_comp[:, 0], recall_comp[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)
              recall_i_comp = data_by_m_b_nrel[["recall", "initial_ncomp"]].sort_values(["recall", "initial_ncomp"], ascending=[True, True])
              recall_i_comp = recall_i_comp[recall_i_comp["recall"].gt(xlim[0])].to_numpy()
              axs[1][i].plot(recall_i_comp[:, 0], recall_i_comp[:, 1] + recall_comp[:, 1], linestyle="--")
              axs[1][i].scatter(recall_i_comp[:, 0], recall_i_comp[:, 1] + recall_comp[:, 1], label=f"total-{m}-{b}-nrel_{nrel}", **marker)

              recall_tqps = data_by_m_b_nrel[["recall", "tqps"]].sort_values(["recall", "tqps"], ascending=[True, False])
              recall_tqps = recall_tqps[recall_tqps["recall"].gt(xlim[0])].to_numpy()
              axs[2][i].plot(recall_tqps[:, 0], recall_tqps[:, 1])
              axs[2][i].scatter(recall_tqps[:, 0], recall_tqps[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)

              recall_prop = data_by_m_b_nrel[["recall", "prop"]].sort_values(["recall", "prop"], ascending=[True, True])
              recall_prop = recall_prop[recall_prop["recall"].gt(xlim[0])].to_numpy()
              axs[3][i].plot(recall_prop[:, 0], recall_prop[:, 1])
              axs[3][i].scatter(recall_prop[:, 0], recall_prop[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)
          else:
            recall_qps = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
            recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
            axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1])
            axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", **marker)
            axs[2][i].plot(recall_qps[:, 0], recall_qps[:, 1])
            axs[2][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", **marker)

            recall_comp = data_by_m_b[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
            recall_comp = recall_comp[recall_comp["recall"].gt(xlim[0])].to_numpy()
            axs[1][i].plot(recall_comp[:, 0], recall_comp[:, 1])
            axs[1][i].scatter(recall_comp[:, 0], recall_comp[:, 1], label=f"{m}-{b}", **marker)

          axs[0][i].set_xlabel('Recall')
          axs[0][i].set_ylabel('QPS')
          axs[0][i].set_title("{}, Selectivity-{:.1%}".format(d.capitalize(), sel))
          axs[1][i].set_xlabel('Recall')
          axs[1][i].set_ylabel('# Comp')
          axs[1][i].set_title("{}, Selectivity-{:.1%}".format(d.capitalize(), sel))
          axs[2][i].set_xlabel('Recall')
          axs[2][i].set_ylabel('Tampered QPS')
          axs[2][i].set_title("{}, Selectivity-{:.1%}".format(d.capitalize(), sel))
          axs[3][i].set_xlabel('Recall')
          axs[3][i].set_ylabel('Initial / Racing')
          axs[3][i].set_title("{}, Selectivity-{:.1%}".format(d.capitalize(), sel))

      fig.set_size_inches(20, 9)
      unique_labels = {}
      for ax in axs.flat:
        ax.set_xlim(xlim)
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
          if label not in unique_labels:
            unique_labels[label] = handle
      fig.legend(unique_labels.values(), unique_labels.keys(), loc="outside right upper")
      path = Path(f"{prefix}/All-{anno}-{rg}-QPS-Comp-Recall.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      fig.savefig(path, dpi=200)
      plt.close()


def draw_qps_comp_fixing_recall_by_dataset_selectivity(da, datasets, methods, anno, *, d_m_b={}, d_m_s={}, prefix="figures"):
  df = pd.read_csv(f"stats-{da}d.csv", dtype=types)
  recall_thresholds = [0.8, 0.9, 0.95]
  selectivities = DA_SEL[da]

  for d in datasets:
    for rec in recall_thresholds:
      fig, axs = plt.subplots(1, 2, layout='constrained')

      data = df[df["dataset"] == d]
      for m in d_m_b[d].keys() if d in d_m_b else methods:
        marker = M_STYLE[m]
        for b in d_m_b.get(d, {}).get(m, data[data["method"] == m].build.unique()):
          data_by_m_b = data[(data["method"] == m) & (data["build"].str.contains(b))]
          if m.startswith("Compass"):
            for nrel in d_m_s.get(d, {}).get(m, {}).get("nrel", compass_args["nrel"]):
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].str.contains(f"nrel_{nrel}")]
              rec_sel_qps_comp = data_by_m_b_nrel[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
              grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
              grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
              pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
              axs[0].plot(pos_s, grouped_qps["qps"])
              axs[0].scatter(pos_s, grouped_qps["qps"], label=f"{m}-{b}-{rec}-{nrel}", **marker)
              axs[1].plot(pos_s, grouped_comp["ncomp"])
              axs[1].scatter(pos_s, grouped_comp["ncomp"], label=f"{m}-{b}-{rec}-{nrel}", **marker)
          else:
            rec_sel_qps_comp = data_by_m_b[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
            grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
            grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
            pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
            axs[0].plot(pos_s, grouped_qps["qps"])
            axs[0].scatter(pos_s, grouped_qps["qps"], label=f"{m}-{b}-{rec}", **marker)
            axs[1].plot(pos_s, grouped_comp["ncomp"])
            axs[1].scatter(pos_s, grouped_comp["ncomp"], label=f"{m}-{b}-{rec}", **marker)

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
      path = Path(f"{prefix}/{d.upper()}/Recall-{rec:.3g}-{anno}-{d.upper()}-QPS-Comp.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      fig.savefig(path, dpi=200)
      plt.close()


def draw_qps_comp_fixing_recall_by_selectivity(da, datasets, methods, anno, *, d_m_b={}, d_m_s={}, prefix="figures"):
  df = pd.read_csv(f"stats-{da}d.csv", dtype=types)
  recall_thresholds = [0.8, 0.9, 0.95]
  selectivities = DA_SEL[da]

  for rec in recall_thresholds:
    fig, axs = plt.subplots(2, len(datasets), layout='constrained')

    for i, d in enumerate(datasets):
      data = df[df["dataset"] == d]
      for m in d_m_b[d].keys() if d in d_m_b else methods:
        marker = M_STYLE[m]
        for b in d_m_b.get(d, {}).get(m, data[data["method"] == m].build.unique()):
          data_by_m_b = data[(data["method"] == m) & (data["build"].str.contains(b))]
          if m.startswith("Compass"):
            for nrel in d_m_s.get(d, {}).get(m, {}).get("nrel", compass_args["nrel"]):
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].str.contains(f"nrel_{nrel}")]
              rec_sel_qps_comp = data_by_m_b_nrel[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
              grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
              grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
              pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
              axs[0][i].plot(pos_s, grouped_qps["qps"])
              axs[0][i].scatter(pos_s, grouped_qps["qps"], label=f"{m}-{b}-{rec}-{nrel}", **marker)
              axs[1][i].plot(pos_s, grouped_comp["ncomp"])
              axs[1][i].scatter(pos_s, grouped_comp["ncomp"], label=f"{m}-{b}-{rec}-{nrel}", **marker)
          else:
            rec_sel_qps_comp = data_by_m_b[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
            grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
            grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
            pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
            axs[0][i].plot(pos_s, grouped_qps["qps"])
            axs[0][i].scatter(pos_s, grouped_qps["qps"], label=f"{m}-{b}-{rec}", **marker)
            axs[1][i].plot(pos_s, grouped_comp["ncomp"])
            axs[1][i].scatter(pos_s, grouped_comp["ncomp"], label=f"{m}-{b}-{rec}", **marker)

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
    path = Path(f"{prefix}/Recall-{rec:.3g}-{anno}-All-QPS-Comp.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close()


if __name__ == "__main__":
  summarize()
  for da in DA_S:
    draw_qps_comp_wrt_recall_by_dataset_selectivity(da, DATASETS, METHODS, "MoM", prefix=f"figures{da}d-10")
    draw_qps_comp_wrt_recall_by_selectivity(da, DATASETS, METHODS, "MoM", prefix=f"figures{da}d-10")
    draw_qps_comp_fixing_recall_by_dataset_selectivity(da, DATASETS, METHODS, "MoM", prefix=f"figures{da}d-10")
    draw_qps_comp_fixing_recall_by_selectivity(da, DATASETS, METHODS, "MoM", prefix=f"figures{da}d-10")
