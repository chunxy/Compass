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
)

nrel_100 = {d: {} for d in DATASETS}
nrel_100_200 = {d: {} for d in DATASETS}
nrel_50_100_200 = {d: {} for d in DATASETS}

for d in DATASETS:
  for m in COMPASS_METHODS:
    nrel_100[d][m] = {"nrel": [100]}
    nrel_100_200[d][m] = {"nrel": [100, 200]}
    nrel_50_100_200[d][m] = {"nrel": [50, 100, 200]}


# Compare clustering methods.
# Choose one clustering method according to these figures,
# potentially one for each dataset.
def pick_clustering_methods():
  clus_methods = ["CompassK", "CompassBikmeans", "CompassPca"]
  d_m_b = {d: {} for d in DATASETS}
  for d in ("sift", "audio"):
    for m in clus_methods:
      d_m_b[d][m] = ["M_16_efc_200_nlist_10000"]
  for d in ("gist", "video", "crawl", "glove100"):
    for m in clus_methods:
      d_m_b[d][m] = ["M_16_efc_200_nlist_20000"]

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
  clus_search_methods = ["CompassK", "CompassKCg", "CompassKIcg"]
  d_m_b = {d: {} for d in DATASETS}
  for d in ("sift", "audio"):
    for m in clus_search_methods:
      d_m_b[d][m] = ["M_16_efc_200_nlist_10000"]
  for d in ("gist", "video", "crawl", "glove100"):
    for m in clus_search_methods:
      d_m_b[d][m] = ["M_16_efc_200_nlist_20000"]

  for da in DA_S:
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=clus_search_methods,
      anno="SoS",
      d_m_b=d_m_b,
      d_m_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-clus-search",
    )
    draw_qps_comp_fixing_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=clus_search_methods,
      anno="SoS",
      d_m_b=d_m_b,
      d_m_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-clus-search",
    )


def pick_nrel():
  methods = ["CompassK"]
  d_m_b = {d: {} for d in DATASETS}
  for d in ("sift", "audio"):
    for m in methods:
      d_m_b[d][m] = ["M_16_efc_200_nlist_10000"]
  for d in ("gist", "video", "crawl", "glove100"):
    for m in methods:
      d_m_b[d][m] = ["M_16_efc_200_nlist_20000"]

  for da in DA_S:
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=methods,
      anno="nrel",
      d_m_b=d_m_b,
      d_m_s=nrel_50_100_200,
      prefix=f"cherrypick{da}d-10/varying-nrel",
    )
    draw_qps_comp_fixing_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=methods,
      anno="nrel",
      d_m_b=d_m_b,
      d_m_s=nrel_50_100_200,
      prefix=f"cherrypick{da}d-10/varying-nrel",
    )


def pick_M():
  methods = ["CompassK"]
  d_m_b_M = {d: {} for d in DATASETS}
  for d in ("sift", "audio"):
    for m in methods:
      d_m_b_M[d][m] = ["M_16_efc_200_nlist_10000", "M_32_efc_200_nlist_10000"]
  for d in ("gist", "video", "crawl", "glove100"):
    for m in methods:
      d_m_b_M[d][m] = ["M_16_efc_200_nlist_20000", "M_32_efc_200_nlist_20000"]

  for da in DA_S:
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=methods,
      anno="M",
      d_m_b=d_m_b_M,
      d_m_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-M",
    )
    draw_qps_comp_fixing_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=methods,
      anno="M",
      d_m_b=d_m_b_M,
      d_m_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-M",
    )


def pick_nlist():
  methods = ["CompassK"]
  d_m_b_nlist = {d: {} for d in DATASETS}
  for d in ("sift", "audio"):
    for m in methods:
      d_m_b_nlist[d][m] = [f"M_16_efc_200_nlist_{nlist}" for nlist in [5000, 10000]]
  for d in ("gist", "video", "crawl", "glove100"):
    for m in methods:
      d_m_b_nlist[d][m] = [f"M_16_efc_200_nlist_{nlist}" for nlist in [10000, 20000]]

  for da in DA_S:
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=methods,
      anno="nlist",
      d_m_b=d_m_b_nlist,
      d_m_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-nlist",
    )
    draw_qps_comp_fixing_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=methods,
      anno="nlist",
      d_m_b=d_m_b_nlist,
      d_m_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-nlist",
    )


# Compare with iRangeGraph and SeRF.
def compare_with_sotas():
  d_m_b = {d: {} for d in DATASETS}
  for d in ("sift", "audio"):
    for m in COMPASS_METHODS:
      d_m_b[d][m] = ["M_16_efc_200_nlist_10000"]
    for m in COMPASSX_METHODS:
      d_m_b[d][m] = [f"M_16_efc_200_nlist_10000_dx_{dx}" for dx in D_ARGS[d]["dx"]]
  for d in ("gist", "video", "crawl", "glove100"):
    for m in COMPASS_METHODS:
      d_m_b[d][m] = ["M_16_efc_200_nlist_20000"]
    for m in COMPASSX_METHODS:
      d_m_b[d][m] = [f"M_16_efc_200_nlist_20000_dx_{dx}" for dx in D_ARGS[d]["dx"]]
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
  d_m_b = {d: {} for d in DATASETS}
  for d in ("sift", "audio"):
    d_m_b[d]["CompassKIcg"] = ["M_16_efc_200_nlist_5000"]
  d_m_b["gist"]["CompassPcaIcg"] = ["M_16_efc_200_nlist_10000_dx_512"]
  d_m_b["video"]["CompassPcaIcg"] = ["M_16_efc_200_nlist_10000_dx_512"]
  d_m_b["crawl"]["CompassPcaIcg"] = ["M_16_efc_200_nlist_10000_dx_128"]
  d_m_b["glove100"]["CompassKIcg"] = ["M_16_efc_200_nlist_10000"]
  for d in DATASETS:
    d_m_b[d]["iRangeGraph"] = ["M_32_efc_200"]
    d_m_b[d]["SeRF"] = ["M_32_efc_200_efmax_500"]

  nrel = {d: {} for d in DATASETS}

  for m in COMPASS_METHODS:
    nrel["sift"][m] = {"nrel": [100]}
    nrel["audio"][m] = {"nrel": [100]}
    nrel["glove100"][m] = {"nrel": [100, 200]}
    nrel["crawl"][m] = {"nrel": [200]}
    nrel["video"][m] = {"nrel": [200]}
    nrel["gist"][m] = {"nrel": [100, 200]}


  for da in DA_S:
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=d_m_b,
      d_m_s=nrel,
      prefix=f"cherrypick{da}d-10/best",
    )
    draw_qps_comp_fixing_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=d_m_b,
      d_m_s=nrel,
      prefix=f"cherrypick{da}d-10/best",
    )


if __name__ == "__main__":
  # pick_clustering_methods()
  # pick_cluster_search_methods()
  # pick_nrel()
  # pick_M()
  # pick_nlist()
  compare_with_sotas()
  compare_best_with_sotas()
