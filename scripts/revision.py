from config import (
  M_ARGS,
  M_PARAM,
  M_STYLE,
  D_ARGS,
)
import json
from pathlib import Path
import pandas as pd
from itertools import product
from matplotlib import pyplot as plt

DATASET_NBASE = {
  "sift-dedup": 1000000 - 14538,
  "audio-dedup": 1000000,
  "gist-dedup": 1000000 - 17306,
  "video-dedup": 1000000,
  "glove100": 1183514,
  "crawl": 1989995,
  "flickr": 4203901,
  "deep10m": 10000000,
}

DATASET_NQUERY = {
  "sift-dedup": 10000,
  "audio-dedup": 10000,
  "gist-dedup": 1000,
  "video-dedup": 10000,
  "glove100": 10000,
  "crawl": 10000,
  "flickr": 29999,
  "deep10m": 10000,
}

DATASET_NDIM = {
  "sift-dedup": 128,
  "audio-dedup": 128,
  "gist-dedup": 960,
  "video-dedup": 1024,
  "glove100": 100,
  "crawl": 300,
  "flickr": 512,
  "deep10m": 96,
}

# DATASET_M = {
#   "sift-dedup": 16,
#   "audio-dedup": 16,
#   "gist-dedup": 16,
#   "video-dedup": 32,
#   "glove100": 32,
#   "crawl": 16,
#   "flickr": 32,
#   "deep10m": 32,
# }


class Datacard:

  def __init__(
    self,
    name,
    base_path,
    query_path,
    attr_path,
    wtype,
    workload,
    dim,
    n_base,
    n_queries,
    n_groundtruth,
    attr_dim,
  ):
    self.name = name
    self.base_path = base_path
    self.query_path = query_path
    self.attr_path = attr_path
    self.wtype = wtype
    self.workload = workload
    self.dim = dim
    self.n_base = n_base
    self.n_queries = n_queries
    self.n_groundtruth = n_groundtruth
    self.attr_dim = attr_dim


BASE = "/home/chunxy/datasets/{}/{}_base.fvecs"
QUERY = "/home/chunxy/datasets/{}/{}_query.fvecs"
ATTR = "/home/chunxy/repos/Compass/data/attr/{}_{}_{}.value.bin"
WORKLOAD = "{}_{}_10_{}"

da_s = (1, 2, 2, 1, 1, 1)
wtypes = ("skewed", "correlated", "anticorrelated", "onesided", "point", "negation")
span_s = (30, 20, 20, 30, 30, 30)

REVISION_CARDS = {
  d: [
    Datacard(
      name=d,
      base_path=BASE.format(d, d),
      query_path=QUERY.format(d, d),
      attr_path=ATTR.format(d, da, 10000),
      wtype=wtype,
      workload=WORKLOAD.format(d, span, wtype),
      dim=DATASET_NDIM[d],
      n_base=DATASET_NBASE[d],
      n_queries=DATASET_NQUERY[d],
      n_groundtruth=100,
      attr_dim=da,
    )
    for da, span, wtype in zip(da_s, span_s, wtypes)
  ]
  for d in DATASET_NBASE.keys()
}
REVISION_CARDS["video-dedup"].append(
  Datacard(
    name="video-dedup",
    base_path=BASE.format("video-dedup", "video-dedup"),
    query_path=QUERY.format("video-dedup", "video-dedup"),
    attr_path=ATTR.format("video-dedup", 2, 10000, "real"),
    wtype="real",
    workload=WORKLOAD.format("video-dedup", 10000, "real"),
    dim=DATASET_NDIM["video-dedup"],
    n_base=DATASET_NBASE["video-dedup"],
    n_queries=DATASET_NQUERY["video-dedup"],
    n_groundtruth=100,
    attr_dim=2,
  )
)

LOG_ROOT = Path("/home/chunxy/repos/Compass/logs_10")

NAVIX_TYPE_NO = {
  "skewed": 0,
  "correlated": 1,
  "onesided": 2,
  "point": 3,
  "negation": 4,
  "anticorrelated": 5,
  "real": 6,
}

types = {
  "path": str,
  "method": str,
  "workload": str,
  "dataset": str,
  "type": str,
  "build": str,
  "search": str,
  "recall": float,
  "qps": float,
  "tqps": float,
  "ncomp": float,
  "prop": float,
  "initial_ncomp": float,
}


def summarize_revision():
  entries = []
  for d in DATASET_NBASE.keys():
    for card in REVISION_CARDS[d]:
      for m in ["SeRF", "SeRF+Post", "CompassPostKTh", "ACORN", "Navix", "Milvus", "Weaviate", "Prefiltering"]:
        w = card.workload if not m.startswith("SeRF") else f"{d}_{10}_{card.wtype}"
        bt = "_".join([f"{bp}_{{}}" for bp in M_PARAM[m]["build"]])
        st = "_".join([f"{sp}_{{}}" for sp in M_PARAM[m]["search"]])
        if m == "CompassGraph":
          ba_s = [[1] if bp == "nlist" else D_ARGS[d].get(bp, M_ARGS[m][bp]) for bp in M_PARAM[m]["build"]]
          sa_s = [D_ARGS[d].get(sp, M_ARGS[m][sp]) for sp in M_PARAM[m]["search"]]
        elif m.startswith("Compass"):
          ba_s = [D_ARGS[d].get(bp, M_ARGS[m][bp]) for bp in M_PARAM[m]["build"]]
          sa_s = [D_ARGS[d].get(sp, M_ARGS[m][sp]) for sp in M_PARAM[m]["search"]]
        else:
          ba_s = [M_ARGS[m][bp] for bp in M_PARAM[m]["build"]]
          sa_s = [M_ARGS[m][sp] for sp in M_PARAM[m]["search"]]
        if d in {"flickr", "deep10m"}:
          ba_s = [D_ARGS[d].get(bp, M_ARGS[m][bp]) for bp in M_PARAM[m]["build"]]
        for ba in product(*ba_s):
          b = bt.format(*ba)
          for sa in product(*sa_s):
            s = st.format(*sa)
            if m == "Navix":
              path = LOG_ROOT / "Navix" / d / "revision" / f"output_{NAVIX_TYPE_NO[card.wtype]}_{sa[0]}_navix.json"
            else:
              path = LOG_ROOT / m / w / b / s
            if path.exists():
              entries.append((path, m, w, d, card.wtype, b, s))
              if (len(entries) % 100 == 0):
                print(f"Processed {len(entries)} entries")

  df = pd.DataFrame.from_records(entries, columns=[
    "path",
    "method",
    "workload",
    "dataset",
    "type",
    "build",
    "search",
  ], index="path")

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
      ncomp.append(stat["aggregated"].get("num_computations", 0))
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
      # Not to use average for now.
      if e[1].startswith("SeRF"):
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

  df.to_csv("stats-revision.csv")


xlim = [0.6, 1]


def draw_qps_comp_wrt_recall_by_workload(datasets, methods, anno, *, d_m_b={}, d_m_s={}, prefix="revision"):
  df_all = pd.read_csv("stats-revision.csv", dtype=types)
  df_all = df_all.fillna('')

  for da, wtype in zip(da_s, wtypes):
    selector = df_all["workload"].str.endswith(f"_{wtype}")
    if not selector.any():
      continue
    df = df_all[selector]
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
            for nrel in d_m_s.get(d, {}).get(m, {}).get("nrel", []):
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].str.contains(f"nrel_{nrel}")]
              recall_qps = data_by_m_b_nrel[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
              recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
              axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
              axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)

              recall_ncomp = data_by_m_b_nrel[["recall", "ncomp", "initial_ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
              recall_ncomp = recall_ncomp[recall_ncomp["recall"].gt(xlim[0])].to_numpy()
              axs[1][i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1] + recall_ncomp[:, 2], **marker)
              axs[1][i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1] + recall_ncomp[:, 2], label=f"{m}-{b}-nrel_{nrel}", **marker)

          else:
            recall_qps = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
            recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
            axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
            axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", **marker)

            recall_ncomp = data_by_m_b[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
            recall_ncomp = recall_ncomp[recall_ncomp["recall"].gt(xlim[0])].to_numpy()
            axs[1][i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1], **marker)
            axs[1][i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1], label=f"{m}-{b}", **marker)

          axs[0][i].set_xlabel('Recall')
          axs[0][i].set_ylabel('QPS')
          axs[0][i].set_title("{}, {}".format(d.capitalize(), wtype.capitalize()))
          axs[1][i].set_xlabel('Recall')
          axs[1][i].set_ylabel('#Comp')
          axs[1][i].set_title("{}, {}".format(d.capitalize(), wtype.capitalize()))

    fig.set_size_inches(20, 6)
    unique_labels = {}
    for ax in axs.flat:
      ax.set_xlim(xlim)
      handles, labels = ax.get_legend_handles_labels()
      for handle, label in zip(handles, labels):
        if label not in unique_labels:
          unique_labels[label] = handle
    fig.legend(unique_labels.values(), unique_labels.keys(), loc="outside right upper")
    path = Path(f"{prefix}/All-{anno}-{wtype}-QPS-Comp-Recall.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close()


def draw_qps_comp_wrt_recall_by_workload_camera(datasets, methods, anno, *, d_m_b={}, d_m_s={}, prefix="camera-ready/revision"):
  df_all = pd.read_csv("stats-revision.csv", dtype=types)
  df_all = df_all.fillna('')

  xlim, xticks = [0.8, 1], [0.8, 0.9, 1]
  dataset_comp_ylim = {
    "crawl": 15000,
    "gist-dedup": 15000,
    "video-dedup": 30000,
    "glove100": 30000,
  }

  selected_efs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 500, 600, 800, 1000]
  selected_efs = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 500, 600, 800, 1000]

  rg_d_b = {}
  for da, wtype in zip(da_s, wtypes):
    selector = df_all["workload"].str.endswith(f"_{wtype}")
    if not selector.any():
      continue
    df = df_all[selector]

    fig, axs = plt.subplots(2, len(datasets), layout='tight')
    fig.set_size_inches(8.5, 4)
    for ax in axs.flat:
      ax.set_box_aspect(1)
      ax.tick_params(axis='y', rotation=45)

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
            for nrel in d_m_s.get(d, {}).get(m, {}).get("nrel", []):
              selected_search = [f"efs_{efs}_nrel_{nrel}_batch_k_20_initial_efs_20_delta_efs_20" for efs in selected_efs]
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].isin(selected_search)]
              recall_qps = data_by_m_b_nrel[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
              recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
              axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
              axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)

              recall_ncomp = data_by_m_b_nrel[["recall", "ncomp", "initial_ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
              recall_ncomp = recall_ncomp[recall_ncomp["recall"].gt(xlim[0])].to_numpy()
              axs[1][i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1] + recall_ncomp[:, 2], **marker)
              axs[1][i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1] + recall_ncomp[:, 2], label=f"{m}-{b}-nrel_{nrel}", **marker)

          else:
            if m != "Prefiltering":
              selected_search = [f"efs_{efs}" for efs in selected_efs]
              data_by_m_b = data_by_m_b[data_by_m_b["search"].isin(selected_search)]
            recall_qps = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
            recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
            axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
            axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", **marker)
            if m in {"Milvus", "Weaviate"}: continue # noqa: E701
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
      axs[0][i].set_title("{}, {}".format(dt, wtype.capitalize() if wtype != "point" else "Equality"))
      axs[1][i].set_xlabel('Recall')
      axs[1][i].set_xticks(xticks)
      if i == 0:
        axs[1][i].set_ylabel('# Comp')
      auto_bottom, auto_top = axs[1][i].get_ylim()
      axs[1][i].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
      # axs[1][i].set_title("{}, {}".format(dt, wtype.capitalize()))

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
      path = Path(f"{prefix}/All-{anno}-{wtype}-QPS-Comp-Recall.jpg")
      path.parent.mkdir(parents=True, exist_ok=True)
      fig.savefig(path, dpi=200)
      plt.close("all")

  with open(f"{prefix}/All-Workloads-{anno}-QPS-Comp-Recall-Search.json", "w") as f:
    json.dump(rg_d_b, f, indent=4)

def draw_qps_comp_wrt_recall_by_large_dataset_camera(datasets, methods, anno, *, d_m_b={}, d_m_s={}, prefix="camera-ready/large"):
  df_all = pd.read_csv("stats-revision.csv", dtype=types)
  df_all = df_all.fillna('')

  xlim, xticks = [0.8, 1], [0.8, 0.9, 1]
  workload_ylim = {
    "onesided": 20000,
    "point": 20000,
    "negation": 20000,
    "skewed": 20000,
    "correlated": 20000,
    "anticorrelated": 20000,
  }

  selected_efs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 500, 600, 800, 1000]
  selected_efs = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 500, 600, 800, 1000]

  rg_d_b = {}
  for d in datasets:
    fig, axs = plt.subplots(2, len(wtypes) // 2, layout='tight')
    fig.set_size_inches(6, 4.5)
    for ax in axs.flat:
      ax.set_box_aspect(1)
      ax.tick_params(axis='y', rotation=45)

    for i, (da, wtype) in enumerate(zip(da_s[:len(wtypes) // 2], wtypes[:len(wtypes) // 2])):
      selector = df_all["workload"].str.endswith(f"_{wtype}")
      if not selector.any():
        continue
      df = df_all[selector]
      data = df[df["dataset"] == d]
      for m in d_m_b[d].keys() if d in d_m_b else methods:
        marker = M_STYLE[m]
        for b in d_m_b.get(d, {}).get(m, data[data["method"] == m].build.unique()):
          if da > 1 and (m == "SeRF" or m == "iRangeGraph"):
            data_by_m_b = data[(data["method"] == m + "+Post") & (data["build"] == b)]
          else:
            data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
          if m.startswith("Compass"):
            for nrel in d_m_s.get(d, {}).get(m, {}).get("nrel", []):
              selected_search = [f"efs_{efs}_nrel_{nrel}_batch_k_20_initial_efs_20_delta_efs_20" for efs in selected_efs]
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].isin(selected_search)]
              recall_qps = data_by_m_b_nrel[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
              recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
              axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
              axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)

              recall_ncomp = data_by_m_b_nrel[["recall", "ncomp", "initial_ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
              recall_ncomp = recall_ncomp[recall_ncomp["recall"].gt(xlim[0])].to_numpy()
              axs[1][i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1] + recall_ncomp[:, 2], **marker)
              axs[1][i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1] + recall_ncomp[:, 2], label=f"{m}-{b}-nrel_{nrel}", **marker)

          else:
            if m != "Prefiltering":
              selected_search = [f"efs_{efs}" for efs in selected_efs]
              data_by_m_b = data_by_m_b[data_by_m_b["search"].isin(selected_search)]
            recall_qps = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
            recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
            axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
            axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", **marker)
            if m in {"Milvus", "Weaviate"}: continue # noqa: E701
            recall_ncomp = data_by_m_b[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
            recall_ncomp = recall_ncomp[recall_ncomp["recall"].gt(xlim[0])].to_numpy()
            axs[1][i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1], **marker)
            axs[1][i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1], label=f"{m}-{b}", **marker)

      # axs[0][i].set_xlabel('Recall')
      axs[0][i].set_xticks(xticks)
      axs[0][i].set_xticklabels([])
      if i == 0:
        axs[0][i].set_ylabel('QPS')
      # axs[0][i].set_title("{}, {}".format(dt, wtype.capitalize()))
      axs[0][i].set_title("{}".format(wtype.capitalize()))
      axs[1][i].set_xlabel('Recall')
      axs[1][i].set_xticks(xticks)
      if i == 0:
        axs[1][i].set_ylabel('# Comp')
      auto_bottom, auto_top = axs[1][i].get_ylim()
      axs[1][i].set_ylim(bottom=-200, top=min(auto_top, workload_ylim[wtype]))
      # axs[1][i].set_title("{}, {}".format(dt, wtype.capitalize()))

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

    # Put a legend below current axis
    fig.legend(
      unique_labels.values(),
      unique_labels.keys(),
      loc='outside lower center',
      bbox_to_anchor=(0.5, 0),
      fancybox=True,
      ncol=(len(unique_labels) + 1) // 2,
    )
    # plt.grid(True)
    path = Path(f"{prefix}/{d}-{anno}-distrib-QPS-Comp-Recall.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close("all")

    fig, axs = plt.subplots(2, len(wtypes) // 2, layout='tight')
    fig.set_size_inches(6, 4.5)
    for ax in axs.flat:
      ax.set_box_aspect(1)
      ax.tick_params(axis='y', rotation=45)

    for i, (da, wtype) in enumerate(zip(da_s[len(wtypes) // 2:], wtypes[len(wtypes) // 2:])):
      selector = df_all["workload"].str.endswith(f"_{wtype}")
      if not selector.any():
        continue
      df = df_all[selector]
      data = df[df["dataset"] == d]
      for m in d_m_b[d].keys() if d in d_m_b else methods:
        marker = M_STYLE[m]
        for b in d_m_b.get(d, {}).get(m, data[data["method"] == m].build.unique()):
          if da > 1 and (m == "SeRF" or m == "iRangeGraph"):
            data_by_m_b = data[(data["method"] == m + "+Post") & (data["build"] == b)]
          else:
            data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
          if m.startswith("Compass"):
            for nrel in d_m_s.get(d, {}).get(m, {}).get("nrel", []):
              selected_search = [f"efs_{efs}_nrel_{nrel}_batch_k_20_initial_efs_20_delta_efs_20" for efs in selected_efs]
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].isin(selected_search)]
              recall_qps = data_by_m_b_nrel[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
              recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
              axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
              axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)

              recall_ncomp = data_by_m_b_nrel[["recall", "ncomp", "initial_ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
              recall_ncomp = recall_ncomp[recall_ncomp["recall"].gt(xlim[0])].to_numpy()
              axs[1][i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1] + recall_ncomp[:, 2], **marker)
              axs[1][i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1] + recall_ncomp[:, 2], label=f"{m}-{b}-nrel_{nrel}", **marker)

          else:
            if m != "Prefiltering":
              selected_search = [f"efs_{efs}" for efs in selected_efs]
              data_by_m_b = data_by_m_b[data_by_m_b["search"].isin(selected_search)]
            recall_qps = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
            recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
            axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
            axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", **marker)
            if m in {"Milvus", "Weaviate"}: continue # noqa: E701
            recall_ncomp = data_by_m_b[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
            recall_ncomp = recall_ncomp[recall_ncomp["recall"].gt(xlim[0])].to_numpy()
            axs[1][i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1], **marker)
            axs[1][i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1], label=f"{m}-{b}", **marker)

      # axs[0][i].set_xlabel('Recall')
      axs[0][i].set_xticks(xticks)
      axs[0][i].set_xticklabels([])
      if i == 0:
        axs[0][i].set_ylabel('QPS')
      # axs[0][i].set_title("{}, {}".format(dt, wtype.capitalize()))
      axs[0][i].set_title("{}".format("Equality" if wtype == "point" else wtype.capitalize()))
      axs[1][i].set_xlabel('Recall')
      axs[1][i].set_xticks(xticks)
      if i == 0:
        axs[1][i].set_ylabel('# Comp')
      auto_bottom, auto_top = axs[1][i].get_ylim()
      axs[1][i].set_ylim(bottom=-200, top=min(auto_top, workload_ylim[wtype]))
      # axs[1][i].set_title("{}, {}".format(dt, wtype.capitalize()))

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

    # Put a legend below current axis
    fig.legend(
      unique_labels.values(),
      unique_labels.keys(),
      loc='outside lower center',
      bbox_to_anchor=(0.5, 0),
      fancybox=True,
      ncol=(len(unique_labels) + 1) // 2,
    )
    # plt.grid(True)
    path = Path(f"{prefix}/{d}-{anno}-workload-QPS-Comp-Recall.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close("all")

  with open(f"{prefix}/All-Large-{anno}-QPS-Comp-Recall-Search.json", "w") as f:
    json.dump(rg_d_b, f, indent=4)

def draw_qps_comp_wrt_recall_by_real_dataset_camera(datasets, methods, anno, *, d_m_b={}, d_m_s={}, prefix="camera-ready/revision"):
  df_all = pd.read_csv("stats-revision.csv", dtype=types)
  df_all = df_all.fillna('')

  xlim, xticks = [0.4, 1], [0.4, 0.6, 0.8, 1]
  dataset_comp_ylim = {
    "video-dedup": 30000,
  }

  selected_efs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 500, 600, 800, 1000]
  selected_efs = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 500, 600, 800, 1000]

  for da, wtype in zip([2], ["real"]):
    selector = df_all["workload"].str.endswith(f"_{wtype}")
    if not selector.any():
      continue
    df = df_all[selector]

    fig, axs = plt.subplots(1, len(datasets) * 2, layout='tight')
    fig.set_size_inches(6.2, 2.4)
    for ax in axs.flat:
      ax.set_box_aspect(1)
      ax.tick_params(axis='y', rotation=45)

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
            for nrel in d_m_s.get(d, {}).get(m, {}).get("nrel", []):
              selected_search = [f"efs_{efs}_nrel_{nrel}_batch_k_20_initial_efs_20_delta_efs_20" for efs in selected_efs]
              data_by_m_b_nrel = data_by_m_b[data_by_m_b["search"].isin(selected_search)]
              recall_qps = data_by_m_b_nrel[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
              recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
              axs[i * 2].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
              axs[i * 2].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}-nrel_{nrel}", **marker)

              recall_ncomp = data_by_m_b_nrel[["recall", "ncomp", "initial_ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
              recall_ncomp = recall_ncomp[recall_ncomp["recall"].gt(xlim[0])].to_numpy()
              axs[i * 2 + 1].plot(recall_ncomp[:, 0], recall_ncomp[:, 1] + recall_ncomp[:, 2], **marker)
              axs[i * 2 + 1].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1] + recall_ncomp[:, 2], label=f"{m}-{b}-nrel_{nrel}", **marker)
          else:
            if m != "Prefiltering":
              selected_search = [f"efs_{efs}" for efs in selected_efs]
              data_by_m_b = data_by_m_b[data_by_m_b["search"].isin(selected_search)]
            recall_qps = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
            recall_qps = recall_qps[recall_qps["recall"].gt(xlim[0])].to_numpy()
            axs[i * 2].plot(recall_qps[:, 0], recall_qps[:, 1], **marker)
            axs[i * 2].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", **marker)
            if m in {"Milvus", "Weaviate"}: continue # noqa: E701
            recall_ncomp = data_by_m_b[["recall", "ncomp"]].sort_values(["recall", "ncomp"], ascending=[True, True])
            recall_ncomp = recall_ncomp[recall_ncomp["recall"].gt(xlim[0])].to_numpy()
            axs[i * 2 + 1].plot(recall_ncomp[:, 0], recall_ncomp[:, 1], **marker)
            axs[i * 2 + 1].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1], label=f"{m}-{b}", **marker)

      dt = d.split("-")[0].upper()
      axs[i * 2].set_xlabel('Recall')
      axs[i * 2].set_xticks(xticks)
      axs[i * 2].set_ylabel('QPS')
      axs[i * 2].set_title("{}, {}".format(dt, wtype.capitalize()))
      axs[i * 2 + 1].set_xlabel('Recall')
      axs[i * 2 + 1].set_xticks(xticks)
      axs[i * 2 + 1].set_ylabel('# Comp')
      auto_bottom, auto_top = axs[i * 2 + 1].get_ylim()
      axs[i * 2 + 1].set_ylim(-200, min(auto_top, dataset_comp_ylim[d]))
      axs[i * 2 + 1].set_title("{}, {}".format(dt, wtype.capitalize()))

    # bottom = 0.15
    plt.tight_layout(rect=[0, 0, 0.8, 1], w_pad=0.1, h_pad=0.05)
    unique_labels = {}
    for ax in axs.flat:
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
    # # Put a legend below current axis
    # fig.legend(
    #   unique_labels.values(),
    #   unique_labels.keys(),
    #   loc='outside lower center',
    #   bbox_to_anchor=(0.5, 0),
    #   fancybox=True,
    #   ncol=(len(unique_labels) + 1) // 2,
    # )
    # Put a legend on right
    fig.legend(
      unique_labels.values(),
      unique_labels.keys(),
      loc='outside right center',
      fancybox=True,
    )
    # plt.grid(True)
    path = Path(f"{prefix}/{d}-{anno}-{wtype}-QPS-Comp-Recall.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close("all")


if __name__ == "__main__":
  summarize_revision()
