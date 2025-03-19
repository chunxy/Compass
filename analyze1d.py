from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import json
import pandas as pd
from scipy.spatial import ConvexHull
import bisect

from config import *

K = 10
LOGS = Path(LOGS_TMPL.format(K))

def summarize_1d():
  entries = [(
    LOGS / m / METHOD_WORKLOAD_TMPL[m].format(d, *rg, K) / METHOD_BUILD_TMPL[m].format(*b) / METHOD_SEARCH_TMPL[m].format(*r),
    m,
    METHOD_WORKLOAD_TMPL[m].format(d, *rg, K),
    d,
    METHOD_BUILD_TMPL[m].format(*b),
    METHOD_SEARCH_TMPL[m].format(*r),
  ) for m in ONED_METHODS for d in DATASETS for rg in METHOD_RANGE_MAPPING[m] for b in METHOD_BUILD_MAPPING[m] for r in METHOD_SEARCH_MAPPING[m]]
  legal_entries = list(filter(lambda e: e[0].exists(), entries))
  df = pd.DataFrame.from_records(legal_entries, columns=["path", "method", "workload", "dataset", "build", "run"], index="path")

  rec, qps, comp, selectivities = [], [], [], []
  for e in legal_entries:
    jsons = list(e[0].glob("*.json"))
    if len(jsons) == 0:
      i = df[(df.index == e[0])].index
      df = df.drop(i)
      continue
    jsons.sort()
    with open(jsons[-1]) as f:
      stat = json.load(f)
      max_rec_prec = stat["aggregated"]["recall"]
      # if stat["aggregated"]["precision"]:
      #   max_rec_prec = max(max_rec_prec, stat["aggregated"]["precision"])
      rec.append(max_rec_prec)
      comp.append(stat["aggregated"]["num_computations"])
      qps.append(stat["aggregated"]["qps"])

    splits = e[2].split("_")
    if len(splits) == 4:
      selectivity = int(splits[1]) / int(splits[2])
    else:
      selectivity = (int(splits[3]) - int(splits[2])) / int(splits[1])
    selectivities.append(str(selectivity))

  df["recall"] = rec
  df["qps"] = qps
  df["comp"] = comp
  df["selectivity"] = selectivities
  df.to_csv(f"stats1d_{K}.csv")


def draw_1d_qps_comp_wrt_recall_by_dataset_selectivity():
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

  selectors = [((df["dataset"] == d) & (df["selectivity"] == r)) for d in DATASETS for r in ONED_PASSRATES]

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
        axs[0].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", marker=METHOD_MARKER_MAPPING[m])
        axs[0].set_xlabel('Recall')
        axs[0].set_ylabel('QPS')
        axs[0].set_title("{}, Selectivity-{:.1%}".format(dataset.capitalize(), selectivity))

        comp_qps = data_by_m_b[["recall", "comp"]].sort_values(["recall", "comp"], ascending=[True, True])
        comp_qps = comp_qps.to_numpy()
        axs[1].plot(comp_qps[:, 0], comp_qps[:, 1])
        axs[1].scatter(comp_qps[:, 0], comp_qps[:, 1], label=f"{m}-{b}", marker=METHOD_MARKER_MAPPING[m])
        axs[1].set_xlabel('Recall')
        axs[1].set_ylabel('# Comp')
        axs[1].set_title("{}, Selectivity-{:.1%}".format(dataset.capitalize(), selectivity))

    # fig.set_size_inches(15, 10)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside right upper')
    fig.savefig(f"figures_{K}/{dataset.upper()}-{selectivity:.1%}-QPS-Comp-Recall.jpg", dpi=200)
    plt.close()


def draw_1d_qps_comp_wrt_recall_by_selectivity():
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
    dataset = data["dataset"].reset_index(drop=True)[0]
    selectivity = float(data["selectivity"].reset_index(drop=True)[0])

    fig, axs = plt.subplots(2, len(DATASETS), layout='constrained')
    for i, dataset in enumerate(DATASETS):
      for m in data.method.unique():
        if m == "Serf": continue
        for b in data[data["method"] == m].build.unique():
          data_by_m_b = data[(data["method"] == m) & (data["build"] == b) & (data["dataset"] == dataset)]
          recall_qps = data_by_m_b[["recall", "qps"]].sort_values(["recall", "qps"], ascending=[True, False])
          recall_qps = recall_qps.to_numpy()
          axs[0][i].plot(recall_qps[:, 0], recall_qps[:, 1])
          axs[0][i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", marker=METHOD_MARKER_MAPPING[m])
          axs[0][i].set_xlabel('Recall')
          axs[0][i].set_ylabel('QPS')
          axs[0][i].set_title("{}, Selectivity-{:.1%}".format(dataset.capitalize(), selectivity))

          comp_qps = data_by_m_b[["recall", "comp"]].sort_values(["recall", "comp"], ascending=[True, True])
          comp_qps = comp_qps.to_numpy()
          axs[1][i].plot(comp_qps[:, 0], comp_qps[:, 1])
          axs[1][i].scatter(comp_qps[:, 0], comp_qps[:, 1], label=f"{m}-{b}", marker=METHOD_MARKER_MAPPING[m])
          axs[1][i].set_xlabel('Recall')
          axs[1][i].set_ylabel('# Comp')
          axs[1][i].set_title("{}, Selectivity-{:.1%}".format(dataset.capitalize(), selectivity))

    fig.set_size_inches(35, 10)
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside right upper')
    fig.savefig(f"figures_{K}/All-{selectivity:.1%}-QPS-Comp-Recall.jpg", dpi=200)
    plt.close()


def draw_1d_by_dataset():
  types = {
    "path": str,
    "method": str,
    "workload": str,
    "build": str,
    "dataset": str,
    "selectivity": float,
    "run": str,
    "recall": float,
    "qps": float,
  }
  df = pd.read_csv(f"stats1d_{K}.csv", dtype=types)

  selectivities = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9]
  for dataset in DATASETS:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.set_yticks(np.arange(len(selectors)))
    # ax.set_yticklabels(selectivities)
    data = df[df["dataset"] == dataset]
    for m in data.method.unique():
      if m != "CompassIvf1d" and m != "CompassGraph1d" and m != "CompassR1d": continue
      rec_qps_sel = data[data["method"] == m][["recall", "qps", "selectivity"]].sort_values(["selectivity", "recall"])
      rec_qps_sel = rec_qps_sel[rec_qps_sel["recall"] > 0.6]
      if len(rec_qps_sel) < 3: continue

      sel_s = rec_qps_sel[["selectivity"]].to_numpy().ravel()
      rec_s = rec_qps_sel[["recall"]].to_numpy().ravel()
      qps_s = rec_qps_sel[["qps"]].to_numpy().ravel()
      pos_s = np.array([bisect.bisect(selectivities, sel) for sel in sel_s]) - 1
      rec_pos = np.concatenate([rec_s[:, np.newaxis], pos_s[:, np.newaxis]], axis=1)
      order = ConvexHull(rec_pos).vertices

      pos_s, rec_s, qps_s = pos_s[order], rec_s[order], qps_s[order]
      # ax.plot_surface(
      #   rec_s,
      #   pos_s,
      #   qps_s,
      #   rstride=1,
      #   cstride=1,
      #   # facecolors=cm.rgb,
      #   linewidth=0,
      #   antialiased=False,
      #   shade=False,
      # )
      ax.plot_trisurf(pos_s, rec_s, qps_s, label=m, antialiased=True)
      # ax.plot(pos_s, rec_s, qps_s, label=m, antialiased=True, linewidth=0)
      # ax.scatter(pos_s, rec_s, qps_s, s=10, antialiased=True)
      ax.stem(pos_s, rec_s, qps_s)
      ax.set_xlabel('Selectivity')
      ax.set_ylabel('Recall')
      ax.set_zlabel('QPS')
      break
    fig.savefig(f"figures_{K}/{dataset.upper()}.jpg", dpi=200)
    plt.close()
    break


def draw_1d_by_dataset_method():
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

  selectivities = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "0.6", "0.7", "0.8", "0.9"]
  for dataset in DATASETS:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_yticks(np.arange(len(selectivities)))
    ax.set_yticklabels(selectivities)
    data = df[df["dataset"] == dataset]
    for m in data.method.unique():
      rec_qps_sel = data[data["method"] == m][["recall", "qps", "selectivity"]].sort_values(["selectivity", "recall"])
      # rec_qps_sel = rec_qps_sel[rec_qps_sel["recall"] > 0.8]
      facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(selectivities)))

      def polygon_under_graph(x, y):
        """
        Construct the vertex list which defines the polygon filling the space under
        the (x, y) line graph. This assumes x is in ascending order.
        """
        return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

      verts = [
        polygon_under_graph(
          rec_qps_sel[rec_qps_sel["selectivity"] == sel]["recall"].to_numpy().ravel(),
          rec_qps_sel[rec_qps_sel["selectivity"] == sel]["qps"].to_numpy().ravel()
        ) for sel in selectivities
      ]
      pos_s = np.array(selectivities).argsort()
      poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
      ax.add_collection3d(poly, zs=range(len(selectivities)), zdir='y')

      ax.set_xlabel('Recall')
      ax.set_ylabel('Selectivity')
      ax.set_zlabel('QPS')
      fig.set_size_inches(10, 10)
      fig.savefig(f"figures_{K}/{dataset.upper()}-{m}.jpg", dpi=200)
      ax.cla()


def draw_1d_qps_comp_fixed_recall_by_dataset_selectivity():
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
  cutoff_recalls = [0.8, 0.9]
  cutoff_colors = ["b", "r"]

  selectivities = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
  for dataset in DATASETS:
    for recall, color in zip(cutoff_recalls, cutoff_colors):
      fig, (ax0, ax1) = plt.subplots(2, layout='constrained', sharex=True)
      ax0.set_xticks(np.arange(len(selectivities)))
      ax0.set_xticklabels(selectivities)
      ax0.set_title(dataset.upper())
      ax0.set_ylabel('QPS')
      ax1.set_ylabel('# Comp')
      ax1.set_xlabel('Selectivity')
      data = df[df["dataset"] == dataset]
      for m in data.method.unique():
        for b in data[data["method"] == m].build.unique():
          data_by_m_b = data[(data["method"] == m) & (data["build"] == b)]
          rec_sel_qps_comp = data_by_m_b[["recall", "selectivity", "qps", "comp"]].sort_values(["selectivity", "recall"])

          grouped_qps = rec_sel_qps_comp[rec_sel_qps_comp["recall"].between(recall - 0.05, recall + 0.05)].groupby("selectivity",
                                                                                                                    as_index=False)["qps"].max()
          grouped_comp = rec_sel_qps_comp[rec_sel_qps_comp["recall"].between(recall - 0.05, recall + 0.05)].groupby("selectivity",
                                                                                                                    as_index=False)["comp"].min()
          pos_s = np.array([bisect.bisect(selectivities, sel) for sel in grouped_qps["selectivity"]]) - 1
          ax0.plot(pos_s, grouped_qps["qps"])
          ax0.scatter(pos_s, grouped_qps["qps"], label=f"{m}-{b}-{recall}", marker=METHOD_MARKER_MAPPING[m])
          ax1.plot(pos_s, grouped_comp["comp"])
          ax1.scatter(pos_s, grouped_comp["comp"], label=f"{m}-{b}-{recall}", marker=METHOD_MARKER_MAPPING[m])

      fig.set_size_inches(15, 10)
      handles, labels = ax0.get_legend_handles_labels()
      fig.legend(handles, labels, loc="outside right upper")
      fig.savefig(f"figures_{K}/{dataset.upper()}-QPS-Comp-Recall-{recall:.1f}.jpg", dpi=200)
      ax0.cla()
      ax1.cla()


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
    fig, axs = plt.subplots(2, 4, layout='constrained')
    fig.set_size_inches(20, 10)
    for i, d in enumerate(SELECTED_DATASETS):
      data = df[df["selectivity"] == passrate]
      data = data[data["dataset"] == d]
      data = data[data["recall"] >= 0.8]
      selectivity = float(passrate)
      for m in SELECTED_BUILDS:
        b = METHOD_BUILD_TMPL[m].format(*SELECTED_BUILDS[m])
        recall_qps = data[(data["method"] == m) & (data["build"] == b)][["recall", "qps"]].sort_values(["recall", "qps"],
                                                                                                        ascending=[True, False]).to_numpy()
        axs[0, i].plot(recall_qps[:, 0], recall_qps[:, 1])
        axs[0, i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", marker=METHOD_MARKER_MAPPING[m])
        axs[0, i].set_xlabel('Recall')
        axs[0, i].set_ylabel('QPS')
        axs[0, i].set_title("{}, Passrate-{:.1%}".format(d.capitalize(), selectivity))

        recall_ncomp = data[(data["method"] == m) & (data["build"] == b)][["recall", "comp"]].sort_values(["recall", "comp"],
                                                                                                          ascending=[True, True]).to_numpy()
        axs[1, i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1])
        axs[1, i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1], label=f"{m}-{b}", marker=METHOD_MARKER_MAPPING[m])
        axs[1, i].set_xlabel('Recall')
        axs[1, i].set_ylabel('#Comp')
        axs[1, i].set_title("{}, Passrate-{:.1%}".format(d.capitalize(), selectivity))
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center")
    fig.savefig(f"figures_{K}/Adverse-Selected-Dataset-{selectivity:.1%}.jpg", dpi=200)
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
    fig, axs = plt.subplots(2, 4, layout='constrained')
    fig.set_size_inches(20, 10)
    for i, d in enumerate(SELECTED_DATASETS):
      data = df[df["selectivity"] == passrate]
      data = data[data["dataset"] == d]
      data = data[data["recall"] >= 0.8]
      selectivity = float(passrate)
      for m in SELECTED_BUILDS:
        b = METHOD_BUILD_TMPL[m].format(*SELECTED_BUILDS[m])
        recall_qps = data[(data["method"] == m) & (data["build"] == b)][["recall", "qps"]].sort_values(["recall", "qps"],
                                                                                                        ascending=[True, False]).to_numpy()
        axs[0, i].plot(recall_qps[:, 0], recall_qps[:, 1])
        axs[0, i].scatter(recall_qps[:, 0], recall_qps[:, 1], label=f"{m}-{b}", marker=METHOD_MARKER_MAPPING[m])
        axs[0, i].set_xlabel('Recall')
        axs[0, i].set_ylabel('QPS')
        axs[0, i].set_title("{}, Passrate-{:.1%}".format(d.capitalize(), selectivity))

        recall_ncomp = data[(data["method"] == m) & (data["build"] == b)][["recall", "comp"]].sort_values(["recall", "comp"],
                                                                                                          ascending=[True, True]).to_numpy()
        axs[1, i].plot(recall_ncomp[:, 0], recall_ncomp[:, 1])
        axs[1, i].scatter(recall_ncomp[:, 0], recall_ncomp[:, 1], label=f"{m}-{b}", marker=METHOD_MARKER_MAPPING[m])
        axs[1, i].set_xlabel('Recall')
        axs[1, i].set_ylabel('#Comp')
        axs[1, i].set_title("{}, Passrate-{:.1%}".format(d.capitalize(), selectivity))
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center")
    fig.savefig(f"figures_{K}/Favorable-Selected-Dataset-{selectivity:.1%}.jpg", dpi=200)
    fig.clf()


plt.rcParams.update({
  'font.size': 15,
  'legend.fontsize': 12,
  'axes.labelsize': 15,
  'axes.titlesize': 15,
  'figure.figsize': (10, 6),
})
# summarize_1d()

draw_1d_qps_comp_wrt_recall_by_dataset_selectivity()
draw_1d_qps_comp_wrt_recall_by_selectivity()
draw_1d_qps_comp_fixed_recall_by_dataset_selectivity()

# draw_1d_by_selected_dataset_adverse()
# draw_1d_by_selected_dataset_favorable()

# draw_1d_by_dataset()
# draw_1d_by_dataset_method()
