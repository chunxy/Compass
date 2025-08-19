from config import (
  COMPASS_METHODS,
  COMPASSX_METHODS,
  DA_S,
  DATASETS,
  METHODS,
  D_ARGS,
)
from summarize import (
  draw_qps_comp_fixing_recall_by_dataset_selectivity,
  draw_qps_comp_fixing_recall_by_selectivity,
  draw_qps_comp_wrt_recall_by_dataset_selectivity,
  draw_qps_comp_wrt_recall_by_selectivity,
  draw_qps_comp_fixing_selectivity_by_dimension_away,
  draw_qps_comp_fixing_selectivity_by_dimension_home,
)

nrel_100 = {d: {} for d in DATASETS}
nrel_50_100 = {d: {} for d in DATASETS}
nrel_100_200 = {d: {} for d in DATASETS}
nrel_50_100_200 = {d: {} for d in DATASETS}

for d in DATASETS:
  for m in COMPASS_METHODS:
    nrel_100[d][m] = {"nrel": [100]}
    nrel_100_200[d][m] = {"nrel": [100, 200]}
    nrel_50_100_200[d][m] = {"nrel": [50, 100, 200]}

best_d_m_b = {d: {} for d in DATASETS}
# for d in ("sift", "audio"):
#   best_d_m_b[d]["CompassBikmeansIcg"] = ["M_16_efc_200_nlist_5000_M_cg_4"]
# best_d_m_b["gist"]["CompassPcaIcg"] = ["M_16_efc_200_nlist_20000_dx_512_M_cg_4"]
# best_d_m_b["gist"]["CompassKIcg"] = ["M_16_efc_200_nlist_10000_M_cg_4"]
# best_d_m_b["video"]["CompassPcaIcg"] = ["M_16_efc_200_nlist_10000_dx_512_M_cg_4"]
# best_d_m_b["video"]["CompassBikmeansIcg"] = ["M_16_efc_200_nlist_20000_M_cg_4"]
# best_d_m_b["crawl"]["CompassPcaIcg"] = ["M_16_efc_200_nlist_20000_dx_128_M_cg_4"]
# best_d_m_b["glove100"]["CompassKIcg"] = ["M_16_efc_200_nlist_10000_M_cg_4"]

for d in ("sift", "audio"):
  best_d_m_b[d]["CompassPostK"] = ["M_16_efc_200_nlist_5000_M_cg_4"]
best_d_m_b["gist"]["CompassPostK"] = ["M_16_efc_200_nlist_10000_M_cg_4"]
best_d_m_b["video"]["CompassPostK"] = ["M_16_efc_200_nlist_10000_M_cg_4"]
best_d_m_b["crawl"]["CompassPostK"] = ["M_16_efc_200_nlist_10000_M_cg_4"]
best_d_m_b["glove100"]["CompassPostK"] = ["M_16_efc_200_nlist_10000_M_cg_4"]
for d in DATASETS:
  best_d_m_b[d]["iRangeGraph"] = ["M_32_efc_200"]
  best_d_m_b[d]["SeRF"] = ["M_32_efc_200_efmax_500"]
  best_d_m_b[d]["Postfiltering"] = ["M_16_efc_200"]
  best_d_m_b[d]["Ivf"] = ["nlist_5000", "nlist_10000"]
  # best_d_m_b[d]["Prefiltering"] = [""]

best_d_m_s = {d: {} for d in DATASETS}
# for m in COMPASS_METHODS:
#   best_d_m_s["sift"][m] = {"nrel": [100]}
#   best_d_m_s["audio"][m] = {"nrel": [100]}
#   best_d_m_s["glove100"][m] = {"nrel": [100]}
#   best_d_m_s["crawl"][m] = {"nrel": [100]}
#   best_d_m_s["video"][m] = {"nrel": [100]}
#   best_d_m_s["gist"][m] = {"nrel": [100]}

for d in DATASETS:
  best_d_m_s[d]["CompassPostK"] = {"nrel": [100]}


# Compare clustering methods.
# Choose one clustering method according to these figures,
# potentially one for each dataset.
def pick_clustering_methods():
  clus_methods = ["CompassKIcg", "CompassBikmeansIcg", "CompassPcaIcg"]
  d_m_b = {d: {} for d in DATASETS}
  for d in ("sift", "audio"):
    for m in clus_methods:
      if m in COMPASSX_METHODS:
        d_m_b[d][m] = [f"M_16_efc_200_nlist_10000_dx_{dx}_M_cg_4" for dx in D_ARGS[d]["dx"]]
      else:
        d_m_b[d][m] = ["M_16_efc_200_nlist_10000_M_cg_4"]
  for d in ("gist", "video", "crawl", "glove100"):
    for m in clus_methods:
      if m in COMPASSX_METHODS:
        d_m_b[d][m] = [f"M_16_efc_200_nlist_10000_dx_{dx}_M_cg_4" for dx in D_ARGS[d]["dx"]]
      else:
        d_m_b[d][m] = ["M_16_efc_200_nlist_10000_M_cg_4"]

  for da in DA_S:
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=clus_methods,
      anno="CoC",
      d_m_b=d_m_b,
      d_m_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-clus",
    )
    draw_qps_comp_fixing_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=clus_methods,
      anno="CoC",
      d_m_b=d_m_b,
      d_m_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-clus",
    )


# Compare cluster search methods.
def pick_cluster_search_methods():
  k_search_methods = ["CompassK", "CompassKCg", "CompassKIcg", "CompassKQicg"]
  pca_search_methods = ["CompassPca", "CompassPcaCg", "CompassPcaIcg", "CompassPcaQicg"]
  d_m_b = {d: {} for d in DATASETS}
  for d in ("sift", "audio"):
    for m in k_search_methods:
      d_m_b[d][m] = ["M_16_efc_200_nlist_10000"]
      if m.endswith("cg") or m.endswith("Cg"):
        for i in range(len(d_m_b[d][m])):
          d_m_b[d][m][i] += "_M_cg_4"
  for m in k_search_methods:
    d_m_b["glove100"][m] = ["M_16_efc_200_nlist_10000"]
    if m.endswith("cg") or m.endswith("Cg"):
      for i in range(len(d_m_b["glove100"][m])):
        d_m_b["glove100"][m][i] += "_M_cg_4"
  for m in pca_search_methods:
    d_m_b["gist"][m] = ["M_16_efc_200_nlist_10000_dx_512"]
    d_m_b["video"][m] = ["M_16_efc_200_nlist_10000_dx_512"]
    d_m_b["crawl"][m] = ["M_16_efc_200_nlist_10000_dx_128"]
    if m.endswith("cg") or m.endswith("Cg"):
      for i in range(len(d_m_b["crawl"][m])):
        d_m_b["crawl"][m][i] += "_M_cg_4"
      for i in range(len(d_m_b["video"][m])):
        d_m_b["video"][m][i] += "_M_cg_4"
      for i in range(len(d_m_b["gist"][m])):
        d_m_b["gist"][m][i] += "_M_cg_4"

  for da in DA_S:
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=k_search_methods,
      anno="SoS",
      d_m_b=d_m_b,
      d_m_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-clus-search",
    )
    draw_qps_comp_fixing_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=k_search_methods,
      anno="SoS",
      d_m_b=d_m_b,
      d_m_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-clus-search",
    )


def pick_nrel():
  methods = ["CompassKIcg", "CompassBikmeansIcg", "CompassPcaIcg"]
  for m in methods:
    d_m_b = {d: {} for d in DATASETS}
    for d in ("sift", "audio"):
      if m != "CompassPcaIcg":
        d_m_b[d][m] = ["M_16_efc_200_nlist_10000_M_cg_4"]
      else:
        d_m_b[d][m] = ["M_16_efc_200_nlist_10000_dx_64_M_cg_4"]
    for d in ("gist", "video", "crawl", "glove100"):
      if m != "CompassPcaIcg":
        d_m_b[d][m] = [f"M_16_efc_200_nlist_{nlist}_M_cg_4" for nlist in [10000, 20000]]
      else:
        dx = 512
        if d == "glove100":
          dx = 64
        elif d == "crawl":
          dx = 128
        d_m_b[d][m] = [f"M_16_efc_200_nlist_{nlist}_dx_{dx}_M_cg_4" for nlist in [10000, 20000]]

    for da in DA_S:
      draw_qps_comp_wrt_recall_by_selectivity(
        da=da,
        datasets=DATASETS,
        methods=methods,
        anno="nrel",
        d_m_b=d_m_b,
        d_m_s=nrel_50_100,
        prefix=f"cherrypick{da}d-10/varying-nrel/{m}",
      )
      draw_qps_comp_fixing_recall_by_selectivity(
        da=da,
        datasets=DATASETS,
        methods=methods,
        anno="nrel",
        d_m_b=d_m_b,
        d_m_s=nrel_50_100_200,
        prefix=f"cherrypick{da}d-10/varying-nrel/{m}",
      )


def pick_M():
  d_m_b_M = {d: {} for d in DATASETS}
  for d in ("sift", "audio"):
    d_m_b_M[d]["CompassKIcg"] = ["M_16_efc_200_nlist_10000_M_cg_4", "M_32_efc_200_nlist_10000_M_cg_4"]
  for d in ("gist", "video"):
    d_m_b_M[d]["CompassPcaIcg"] = ["M_16_efc_200_nlist_20000_dx_512_M_cg_4", "M_32_efc_200_nlist_20000_dx_512_M_cg_4"]
  d_m_b_M["crawl"]["CompassPcaIcg"] = ["M_16_efc_200_nlist_20000_dx_128_M_cg_4", "M_32_efc_200_nlist_20000_dx_128_M_cg_4"]
  d_m_b_M["glove100"]["CompassKIcg"] = ["M_16_efc_200_nlist_20000_M_cg_4", "M_32_efc_200_nlist_20000_M_cg_4"]

  for da in DA_S:
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=[],
      anno="M",
      d_m_b=d_m_b_M,
      d_m_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-M",
    )
    draw_qps_comp_fixing_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=[],
      anno="M",
      d_m_b=d_m_b_M,
      d_m_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-M",
    )


def pick_nlist():
  methods = ["CompassK", "CompassKIcg", "CompassBikmeans", "CompassBikmeansIcg", "CompassPca", "CompassPcaIcg"]
  for m in methods:
    d_m_b_nlist = {d: {} for d in DATASETS}
    for d in ("sift", "audio"):
      if "Pca" in m:
        d_m_b_nlist[d][m] = [(f"M_16_efc_200_nlist_{nlist}_dx_64" + ("_M_cg_4" if m.endswith("cg") else "")) for nlist in [1000, 2000, 5000, 10000]]
      else:
        d_m_b_nlist[d][m] = [(f"M_16_efc_200_nlist_{nlist}" + ("_M_cg_4" if m.endswith("cg") else "")) for nlist in [1000, 2000, 5000, 10000]]
    for d in ("gist", "video", "crawl", "glove100"):
      if "Pca" in m:
        dx = 512
        if d == "glove100":
          dx = 64
        elif d == "crawl":
          dx = 128
        d_m_b_nlist[d][m] = [(f"M_16_efc_200_nlist_{nlist}_dx_{dx}" + ("_M_cg_4" if m.endswith("cg") else "")) for nlist in [1000, 2000, 5000, 10000, 20000]]
      else:
        d_m_b_nlist[d][m] = [(f"M_16_efc_200_nlist_{nlist}" + ("_M_cg_4" if m.endswith("cg") else "")) for nlist in [1000, 2000, 5000, 10000, 20000]]

    for da in DA_S:
      draw_qps_comp_wrt_recall_by_selectivity(
        da=da,
        datasets=DATASETS,
        methods=methods,
        anno="nlist",
        d_m_b=d_m_b_nlist,
        d_m_s=nrel_100,
        prefix=f"cherrypick{da}d-10/varying-nlist/{m}",
      )
      draw_qps_comp_fixing_recall_by_selectivity(
        da=da,
        datasets=DATASETS,
        methods=methods,
        anno="nlist",
        d_m_b=d_m_b_nlist,
        d_m_s=nrel_100,
        prefix=f"cherrypick{da}d-10/varying-nlist/{m}",
      )


def pick_dx():
  methods = ["CompassPca", "CompassPcaIcg"]
  for m in methods:
    d_m_b_dx = {d: {} for d in DATASETS}
    for d in ("gist", "video"):
      d_m_b_dx[d][m] = [(f"M_16_efc_200_nlist_10000_dx_{dx}" + ("_M_cg_4" if m.endswith("cg") else "")) for dx in [256, 512]]
    d_m_b_dx["crawl"][m] = [(f"M_16_efc_200_nlist_20000_dx_{dx}" + ("_M_cg_4" if m.endswith("cg") else "")) for dx in [128, 256]]

    for da in DA_S:
      draw_qps_comp_wrt_recall_by_selectivity(
        da=da,
        datasets=DATASETS,
        methods=methods,
        anno="dx",
        d_m_b=d_m_b_dx,
        d_m_s=nrel_100,
        prefix=f"cherrypick{da}d-10/varying-dx/{m}",
      )
      draw_qps_comp_fixing_recall_by_selectivity(
        da=da,
        datasets=DATASETS,
        methods=methods,
        anno="dx",
        d_m_b=d_m_b_dx,
        d_m_s=nrel_100,
        prefix=f"cherrypick{da}d-10/varying-dx/{m}",
      )


# Compare with iRangeGraph and SeRF.
def compare_with_sotas():
  d_m_b = {d: {} for d in DATASETS}
  for d in ("sift", "audio"):
    for m in COMPASS_METHODS:
      if m.endswith("Qicg") or m.endswith("Cg"):
        continue
      d_m_b[d][m] = [f"M_16_efc_200_nlist_{nlist}" for nlist in [5000, 10000]]
      if m in COMPASSX_METHODS:
        d_m_b[d][m] = [f"M_16_efc_200_nlist_{nlist}_dx_{dx}" for dx in D_ARGS[d]["dx"] for nlist in [5000, 10000]]
      if m.endswith("cg") or m.endswith("Cg"):
        for i in range(len(d_m_b[d][m])):
          d_m_b[d][m][i] += "_M_cg_4"
  for d in ("gist", "video", "crawl", "glove100"):
    for m in COMPASS_METHODS:
      if m.endswith("Qicg") or m.endswith("Cg"):
        continue
      d_m_b[d][m] = [f"M_16_efc_200_nlist_{nlist}" for nlist in [10000, 20000]]
      if m in COMPASSX_METHODS:
        dx = 512 if d != "crawl" else 128
        d_m_b[d][m] = [f"M_16_efc_200_nlist_{nlist}_dx_{dx}" for nlist in [10000, 20000]]
      if m.endswith("cg") or m.endswith("Cg"):
        for i in range(len(d_m_b[d][m])):
          d_m_b[d][m][i] += "_M_cg_4"
  for d in DATASETS:
    d_m_b[d]["iRangeGraph"] = ["M_32_efc_200"]
    d_m_b[d]["SeRF"] = ["M_32_efc_200_efmax_500"]

  for da in DA_S:
    draw_qps_comp_wrt_recall_by_dataset_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=d_m_b,
      d_m_s=nrel_100,
      prefix=f"cherrypick{da}d-10",
    )
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=d_m_b,
      d_m_s=nrel_100,
      prefix=f"cherrypick{da}d-10",
    )
    draw_qps_comp_fixing_recall_by_dataset_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=d_m_b,
      d_m_s=nrel_100,
      prefix=f"cherrypick{da}d-10",
    )
    draw_qps_comp_fixing_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=d_m_b,
      d_m_s=nrel_100,
      prefix=f"cherrypick{da}d-10",
    )


def compare_best_with_sotas():
  for da in DA_S:
    # draw_qps_comp_wrt_recall_by_selectivity(
    #   da=da,
    #   datasets=DATASETS,
    #   methods=METHODS,
    #   anno="MoM",
    #   d_m_b=best_d_m_b,
    #   d_m_s=best_d_m_s,
    #   prefix=f"cherrypick{da}d-10/best",
    # )
    draw_qps_comp_fixing_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=best_d_m_b,
      d_m_s=best_d_m_s,
      prefix=f"cherrypick{da}d-10/best",
    )


def compare_best_with_sotas_by_dimension():
  draw_qps_comp_fixing_selectivity_by_dimension_away(best_d_m_b, best_d_m_s, "dim", "by-dimension-away")
  draw_qps_comp_fixing_selectivity_by_dimension_home(best_d_m_b, best_d_m_s, "dim", "by-dimension-home")


if __name__ == "__main__":
  # pick_clustering_methods()
  # pick_cluster_search_methods()
  # pick_nrel()
  # pick_M()
  # pick_nlist()
  # pick_dx()
  # compare_best_with_sotas_by_dimension()
  # compare_with_sotas() # slow?
  compare_best_with_sotas()
