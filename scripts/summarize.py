import bisect
import json
from functools import reduce
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import (
  BASE_METHODS,
  COMPASS_METHODS,
  COMPASSX_METHODS,
  SOTA_METHODS,
  SOTA_POST_METHODS,
  DA_RANGE,
  DA_S,
  DA_SEL,
  DATASETS,
  M_ARGS,
  M_DA_RUN,
  M_STYLE,
  M_PARAM,
  M_WORKLOAD,
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
dataset_comp_ylim = {
  "sift-dedup": 6000,
  "audio-dedup": 6000,
  "crawl": 6000,
  "gist-dedup": 6000,
  "video-dedup": 20000,
  "glove100": 20000,
}


def summarize():
  for da in [1, 2, 3, 4]:
    entries = []
    for m in SOTA_POST_METHODS + BASE_METHODS + SOTA_METHODS:
      if da not in M_DA_RUN[m]: continue  # noqa: E701
      for d in DATASETS:
        for itvl in M_DA_RUN[m][da]:
          if m == "CompassPostKThCh":
            w = M_WORKLOAD[m].format(
              d,
              itvl[0],
              "-".join(map(str, range(200, 200 + 100 * (da - 1), 100))),
              "-".join(map(str, range(200 + 100 * itvl[0], 200 + 100 * (da - 1) + 100 * itvl[0], 100))),
            )
            nrg = "-".join(map(str, [itvl[0] for _ in range(da)]))
            sel = f"{(itvl[0] ** da) / (100 ** da):.4g}"
          elif m == "ACORN":
            w = M_WORKLOAD[m].format(d, itvl)
            nrg = str(100 / itvl)
            sel = f"{(1 / itvl):.4g}"
          elif m in COMPASS_METHODS or m in BASE_METHODS:
            w = M_WORKLOAD[m].format(d, *map(lambda ele: "-".join(map(str, ele)), itvl))
            nrg = "-".join([f"{(r - l) // 100}" for l, r in zip(*itvl)])  # noqa: E741
            sel = f"{reduce(lambda a, b: a * b, [(r - l) / 10000 for l, r in zip(*itvl)], 1.):.4g}"  # noqa: E741
          else:
            w = M_WORKLOAD[m].format(d, "-".join(map(str, itvl)))
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
                  path = LOG_ROOT / "Navix" / d / f"output_{nrg}_{sa[0]}_navix.json"
                else:
                  path = LOG_ROOT / "Navix" / d / f"{da}d" / f"output_{int(float(sel) * 100)}_{sa[0]}_navix.json"
              else:
                path = LOG_ROOT / m / w / b / s
              if path.exists():
                entries.append((path, m, w, d, nrg, sel, b, s))
                if (len(entries) % 100 == 0):
                  print(f"Processed {len(entries)} entries")

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
      if e[1] in SOTA_METHODS or e[1] in SOTA_POST_METHODS:
        nsample, averaged_qps = min(len(jsons), 3), qps[-1]
        for i in range(2, nsample + 1):
          with open(jsons[-i]) as f:
            stat = json.load(f)
            averaged_qps += stat["aggregated"]["qps"]
        averaged_qps /= nsample
        qps[-1] = averaged_qps
      elif e[1].startswith("CompassPost"):
        nsample, max_qps = min(len(jsons), 3), qps[-1]
        for i in range(2, nsample + 1):
          with open(jsons[-i]) as f:
            stat = json.load(f)
            max_qps = max(max_qps, stat["aggregated"]["qps"])
        qps[-1] = max_qps

    df["recall"] = rec
    df["qps"] = qps
    df["tqps"] = tqps
    df["ncomp"] = ncomp
    df["prop"] = prop
    df["initial_ncomp"] = initial_ncomp

    df.to_csv(f"stats-{da}d.csv")


def draw_qps_comp_wrt_recall_by_dataset_selectivity(da, datasets, methods, anno, *, d_m_b={}, d_m_s={}, prefix="figures"):
  df = pd.read_csv(f"stats-{da}d.csv", dtype=types)
  df = df.fillna('')

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
          if da > 1 and (m == "SeRF" or m == "iRangeGraph"):
            data_by_m_b = data[(data["method"] == m + "+Post") & (data["build"] == b)]
          else:
            data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
          if m.startswith("Compass"):
            for nrel in d_m_s.get(d, {}).get(m, {}).get("nrel", compass_args["nrel"]):
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].str.contains(f"nrel_{nrel}")]
              recall_qps = data_by_m_b_nrel[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
              recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
              axs[0].plot(recall_qps[:, 0], recall_qps[:, 1])
              axs[0].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)

              recall_ncomp = data_by_m_b_nrel[["recall", "ncomp", "initial_ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
              recall_ncomp = recall_ncomp[recall_ncomp["recall"].gt(xlim[0])].to_numpy()
              p = axs[1].plot(recall_ncomp[:, 0], recall_ncomp[:, 1])
              axs[1].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)
              axs[1].plot(recall_ncomp[:, 0], recall_ncomp[:, 1] + recall_ncomp[:, 2], color=p[0].get_color(), linestyle="--")
              axs[1].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1] + recall_ncomp[:, 2], **marker)

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
            axs[0].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
            axs[0].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", **marker)
            axs[2].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
            axs[2].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", **marker)

            recall_ncomp = data_by_m_b[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
            recall_ncomp = recall_ncomp[recall_ncomp["recall"].gt(xlim[0])].to_numpy()
            axs[1].plot(recall_ncomp[:, 0], recall_ncomp[:, 1], **marker)
            axs[1].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1], label=f"{m}-{b}", **marker)

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
  df = df.fillna('')

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
          if da > 1 and (m == "SeRF" or m == "iRangeGraph"):
            data_by_m_b = data[(data["method"] == m + "+Post") & (data["build"] == b)]
          else:
            data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
          if m.startswith("Compass"):
            for nrel in d_m_s.get(d, {}).get(m, {}).get("nrel", compass_args["nrel"]):
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].str.contains(f"nrel_{nrel}")]
              recall_qps = data_by_m_b_nrel[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
              recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
              axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1])
              axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)

              recall_ncomp = data_by_m_b_nrel[["recall", "ncomp", "initial_ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
              recall_ncomp = recall_ncomp[recall_ncomp["recall"].gt(xlim[0])].to_numpy()
              p = axs[1][i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1])
              axs[1][i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)
              axs[1][i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1] + recall_ncomp[:, 2], color=p[0].get_color(), linestyle="--")
              axs[1][i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1] + recall_ncomp[:, 2], **marker)

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
            axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
            axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", **marker)
            axs[2][i].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
            axs[2][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", **marker)

            recall_ncomp = data_by_m_b[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
            recall_ncomp = recall_ncomp[recall_ncomp["recall"].gt(xlim[0])].to_numpy()
            axs[1][i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1], **marker)
            axs[1][i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1], label=f"{m}-{b}", **marker)

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


def draw_qps_comp_wrt_recall_by_selectivity_camera(da, datasets, methods, anno, *, d_m_b={}, d_m_s={}, prefix="figures"):
  xlim = [0.8, 1]
  df = pd.read_csv(f"stats-{da}d.csv", dtype=types)
  df = df.fillna('')
  dataset_comp_ylim = {
    "crawl": 10000,
    "gist-dedup": 10000,
    "video-dedup": 30000,
    "glove100": 30000,
  }

  for rg in DA_RANGE[da]:
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


def draw_qps_comp_fixing_recall_by_dataset_selectivity(da, datasets, methods, anno, *, d_m_b={}, d_m_s={}, prefix="figures"):
  df = pd.read_csv(f"stats-{da}d.csv", dtype=types)
  df = df.fillna('')
  recall_thresholds = [0.8, 0.9, 0.95]
  selectivities = DA_SEL[da]

  for d in datasets:
    for rec in recall_thresholds:
      fig, axs = plt.subplots(1, 2, layout='constrained')

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
              axs[0].plot(pos_s, grouped_qps["qps"])
              axs[0].scatter(pos_s, grouped_qps["qps"], label=f"{m}-{b}-{rec}-{nrel}", **marker)
              p = axs[1].plot(pos_s, grouped_ncomp["ncomp"])
              axs[1].scatter(pos_s, grouped_ncomp["ncomp"], label=f"{m}-{b}-{rec}-{nrel}", **marker)
              axs[1].plot(pos_s, grouped_total_ncomp["total_ncomp"], color=p[0].get_color(), linestyle="--")
              axs[1].scatter(pos_s, grouped_total_ncomp["total_ncomp"], **marker)
          else:
            rec_sel_qps_ncomp = data_by_m_b[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
            grouped_qps = rec_sel_qps_ncomp[rec_sel_qps_ncomp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
            grouped_ncomp = rec_sel_qps_ncomp[rec_sel_qps_ncomp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
            pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
            axs[0].plot(pos_s, grouped_qps["qps"], **marker)
            axs[0].scatter(pos_s, grouped_qps["qps"], label=f"{m}-{b}-{rec}", **marker)
            axs[1].plot(pos_s, grouped_ncomp["ncomp"], **marker)
            axs[1].scatter(pos_s, grouped_ncomp["ncomp"], label=f"{m}-{b}-{rec}", **marker)

      axs[0].set_xticks(np.arange(len(selectivities)))
      axs[0].set_xticklabels(selectivities)
      axs[0].set_xlabel('Selectivity')
      axs[0].set_ylabel('QPS')
      axs[0].set_title(f"{d.upper()}, Recall-{rec:.3g}")
      axs[1].set_xticks(np.arange(len(selectivities)))
      axs[1].set_xticklabels(selectivities)
      axs[1].set_xlabel('Selectivity')
      axs[1].set_ylabel('# Comp')
      auto_bottom, auto_top = axs[1].get_ylim()
      axs[1].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
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
  df = df.fillna('')
  recall_thresholds = [0.8, 0.9, 0.95]
  selectivities = DA_SEL[da]

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
              axs[1][i].plot(pos_s, grouped_ncomp["ncomp"])
              axs[1][i].scatter(pos_s, grouped_ncomp["ncomp"], label=f"{m}-{b}-{rec}-{nrel}", **marker)
              axs[1][i].plot(pos_s, grouped_total_ncomp["total_ncomp"], color=p[0].get_color(), linestyle="--")
              axs[1][i].scatter(pos_s, grouped_total_ncomp["total_ncomp"], **marker)
          else:
            rec_sel_qps_ncomp = data_by_m_b[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
            grouped_qps = rec_sel_qps_ncomp[rec_sel_qps_ncomp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
            grouped_ncomp = rec_sel_qps_ncomp[rec_sel_qps_ncomp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
            pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
            axs[0][i].plot(pos_s, grouped_qps["qps"], **marker)
            axs[0][i].scatter(pos_s, grouped_qps["qps"], label=f"{m}-{b}-{rec}", **marker)
            axs[1][i].plot(pos_s, grouped_ncomp["ncomp"], **marker)
            axs[1][i].scatter(pos_s, grouped_ncomp["ncomp"], label=f"{m}-{b}-{rec}", **marker)

      axs[0][i].set_xticks(np.arange(len(selectivities)))
      axs[0][i].set_xticklabels(selectivities)
      axs[0][i].set_xlabel('Selectivity')
      axs[0][i].set_ylabel('QPS')
      axs[0][i].set_title(f"{d.upper()}, Recall-{rec:.3g}")
      axs[1][i].set_xticks(np.arange(len(selectivities)))
      axs[1][i].set_xticklabels(selectivities)
      axs[1][i].set_xlabel('Selectivity')
      axs[1][i].set_ylabel('# Comp')
      auto_bottom, auto_top = axs[1][i].get_ylim()
      axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
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


def draw_qps_comp_fixing_recall_by_selectivity_camera(da, datasets, methods, anno, *, d_m_b={}, d_m_s={}, prefix="figures"):
  df = pd.read_csv(f"stats-{da}d.csv", dtype=types)
  df = df.fillna('')
  recall_thresholds = [0.9, 0.95]
  selectivities = DA_SEL[da]

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


def draw_qps_comp_fixing_overall_selectivity_by_dimension(datasets, d_m_b, d_m_s, anno, prefix):
  sel_s = [0.01, 0.1, 0.2, 0.4, 0.6]
  interval = {
    0.01: ["0.01", "0.01", "0.008", "0.0081"],
    0.1: ["0.1", "0.09", "0.0911", "0.0915"],
    0.2: ["0.2", "0.2025", "0.216", "0.2401"],
    0.4: ["0.4", "0.4225", "0.4219", "0.4096"],
    0.6: ["0.6", "0.64", "0.6141", "0.6561"],
  }

  for rec in [0.8, 0.9, 0.95]:
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
                  d_m_sel[d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else 20000)
                  d_m_sel[d][m][sel][label]["total_ncomp"].append(grouped_total_comp["total_ncomp"][pos] if pos >= 0 else 20000)
              else:
                rec_sel_qps_comp = data_by_m_b[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
                grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
                grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
                pos = bisect.bisect(grouped_qps["selectivity"], interval[sel][da - 1]) - 1
                if pos == -1 or grouped_qps["selectivity"][pos] != interval[sel][da - 1]:
                  pos = -1
                label = f"{m}-{b}-{rec}"
                if label not in d_m_sel[d][m][sel]:
                  d_m_sel[d][m][sel][label] = {"qps": [], "ncomp": []}
                d_m_sel[d][m][sel][label]["qps"].append(grouped_qps["qps"][pos] if pos >= 0 else 0)
                d_m_sel[d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else 20000)

    for sel in sel_s:
      fig, axs = plt.subplots(2, len(datasets), layout='constrained')
      for i, d in enumerate(datasets):
        for m in d_m_b[d].keys():
          marker = M_STYLE[m]
          for label in d_m_sel[d][m][sel].keys():
            das = DA_S[:len(d_m_sel[d][m][sel][label]["qps"])]
            sc = axs[0][i].scatter(das, d_m_sel[d][m][sel][label]["qps"], label=label, **marker)
            axs[0][i].plot(das, d_m_sel[d][m][sel][label]["qps"], color=sc.get_facecolor()[0])
            axs[1][i].scatter(das, d_m_sel[d][m][sel][label]["ncomp"], label=label, **marker)
            axs[1][i].plot(das, d_m_sel[d][m][sel][label]["ncomp"], color=sc.get_facecolor()[0])
            if m.startswith("Compass"):
              axs[1][i].scatter(das, d_m_sel[d][m][sel][label]["total_ncomp"], label=label, **marker)
              axs[1][i].plot(das, d_m_sel[d][m][sel][label]["total_ncomp"], color=sc.get_facecolor()[0], linestyle="--")

        axs[0][i].set_xlabel('Dimension')
        axs[0][i].set_xticks(DA_S)
        axs[0][i].set_ylabel('QPS')
        axs[0][i].set_title(f"{d.upper()}, Recall-{rec:.3g}")
        axs[1][i].set_xlabel('Dimension')
        axs[1][i].set_ylabel('# Comp')
        axs[1][i].set_xticks(DA_S)
        auto_bottom, auto_top = axs[1][i].get_ylim()
        axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
        axs[1][i].set_title(f"{d.upper()}, Recall-{rec:.3g}")

      fig.set_size_inches(26, 7)
      unique_labels = {}
      for ax in axs.flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
          if label not in unique_labels:
            unique_labels[label] = handle
      fig.legend(unique_labels.values(), unique_labels.keys(), loc="outside right upper")
      path = Path(f"{prefix}/Sel-{sel:.3g}-Recall-{rec:.3g}-{anno}-All-QPS-Comp.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      fig.savefig(path, dpi=200)
      plt.close()


def draw_qps_comp_fixing_dimension_selectivity_by_dimension(datasets, d_m_b, d_m_s, anno, prefix):
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

  for rec in [0.8, 0.9, 0.95]:
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
                  d_m_sel[d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else 20000)
                  d_m_sel[d][m][sel][label]["total_ncomp"].append(grouped_total_comp["total_ncomp"][pos] if pos >= 0 else 20000)
              else:
                rec_sel_qps_comp = data_by_m_b[["recall", "selectivity", "qps", "ncomp"]].sort_values(["selectivity", "recall"])
                grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["qps"].max()
                grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].gt(rec)].groupby("selectivity", as_index=False)["ncomp"].min()
                pos = bisect.bisect(grouped_qps["selectivity"], interval[sel][da - 1]) - 1
                if pos == -1 or grouped_qps["selectivity"][pos] != interval[sel][da - 1]:
                  pos = -1
                label = f"{m}-{b}-{rec}"
                if label not in d_m_sel[d][m][sel]:
                  d_m_sel[d][m][sel][label] = {"qps": [], "ncomp": []}
                d_m_sel[d][m][sel][label]["qps"].append(grouped_qps["qps"][pos] if pos >= 0 else 0)
                d_m_sel[d][m][sel][label]["ncomp"].append(grouped_comp["ncomp"][pos] if pos >= 0 else 20000)

    for sel in sel_s:
      fig, axs = plt.subplots(2, len(datasets), layout='constrained')
      for i, d in enumerate(datasets):
        for m in d_m_b[d].keys():
          marker = M_STYLE[m]
          for label in d_m_sel[d][m][sel].keys():
            das = DA_S[:len(d_m_sel[d][m][sel][label]["qps"])]
            sc = axs[0][i].scatter(das, d_m_sel[d][m][sel][label]["qps"], label=label, **marker)
            axs[0][i].plot(das, d_m_sel[d][m][sel][label]["qps"], color=sc.get_facecolor()[0])
            axs[1][i].scatter(das, d_m_sel[d][m][sel][label]["ncomp"], label=label, **marker)
            axs[1][i].plot(das, d_m_sel[d][m][sel][label]["ncomp"], color=sc.get_facecolor()[0])
            if m.startswith("Compass"):
              axs[1][i].scatter(das, d_m_sel[d][m][sel][label]["total_ncomp"], label=label, **marker)
              axs[1][i].plot(das, d_m_sel[d][m][sel][label]["total_ncomp"], color=sc.get_facecolor()[0], linestyle="--")

        axs[0][i].set_xlabel('Dimension')
        axs[0][i].set_xticks(DA_S)
        axs[0][i].set_ylabel('QPS')
        axs[0][i].set_title(f"{d.upper()}, Recall-{rec:.3g}")
        axs[1][i].set_xlabel('Dimension')
        axs[1][i].set_ylabel('# Comp')
        auto_bottom, auto_top = axs[1][i].get_ylim()
        axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
        axs[1][i].set_xticks(DA_S)
        axs[1][i].set_title(f"{d.upper()}, Recall-{rec:.3g}")

      fig.set_size_inches(26, 7)
      unique_labels = {}
      for ax in axs.flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
          if label not in unique_labels:
            unique_labels[label] = handle
      fig.legend(unique_labels.values(), unique_labels.keys(), loc="outside right upper")
      path = Path(f"{prefix}/Sel-{sel:.3g}-Recall-{rec:.3g}-{anno}-All-QPS-Comp.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
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
    0.3: ["0.3", "0.6", "0.9"],
  }
  dataset_comp_ylim = {
    "crawl": 10000,
    "gist-dedup": 10000,
    "video-dedup": 30000,
    "glove100": 30000,
  }
  ndisjunctions = [1, 2, 3]

  for rec in [0.8, 0.85, 0.9, 0.95]:
    d_m = {}
    for d in d_m_b.keys():
      d_m[d] = {}
      for m in d_m_b[d].keys():
        d_m[d][m] = {}
        for sel in sel_s:
          d_m[d][m][sel] = {}

    for ndis in ndisjunctions:
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


if __name__ == "__main__":
  # pass  # do nothing first
  summarize()
  # for da in DA_S:
  #   draw_qps_comp_wrt_recall_by_dataset_selectivity(da, DATASETS, METHODS, "MoM", prefix=f"figures{da}d-10")
  #   draw_qps_comp_wrt_recall_by_selectivity(da, DATASETS, METHODS, "MoM", prefix=f"figures{da}d-10")
  #   draw_qps_comp_fixing_recall_by_dataset_selectivity(da, DATASETS, METHODS, "MoM", prefix=f"figures{da}d-10")
  #   draw_qps_comp_fixing_recall_by_selectivity(da, DATASETS, METHODS, "MoM", prefix=f"figures{da}d-10")
