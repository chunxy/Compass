from config import (
  COMPASS_METHODS,
  COMPASSX_METHODS,
  DA_S,
  DATASETS,
  METHODS,
  dataset_args,
)
from summarize import (
  draw_qps_comp_fixing_recall_by_dataset_selectivity,
  draw_qps_comp_fixing_recall_by_selectivity,
  draw_qps_comp_wrt_recall_by_dataset_selectivity,
  draw_qps_comp_wrt_recall_by_selectivity,
)

d_m_b_M32 = {d: {} for d in DATASETS}
for d in ("sift", "audio"):
  for m in COMPASS_METHODS:
    d_m_b_M32[d][m] = ["M_16_efc_200_nlist_10000"]
  for m in COMPASSX_METHODS:
    d_m_b_M32[d][m] = [f"M_16_efc_200_nlist_10000_dx_{dx}" for dx in dataset_args[d]["dx"]]
for d in ("gist", "video", "crawl", "glove100"):
  for m in COMPASS_METHODS:
    d_m_b_M32[d][m] = ["M_16_efc_200_nlist_20000"]
  for m in COMPASSX_METHODS:
    d_m_b_M32[d][m] = [f"M_16_efc_200_nlist_20000_dx_{dx}" for dx in dataset_args[d]["dx"]]
for d in DATASETS:
  d_m_b_M32[d]["iRangeGraph"] = ["M_32_efc_200"]
  d_m_b_M32[d]["SeRF"] = ["M_32_efc_200_efmax_500"]

nrel_100 = [100]
nrel_100_200 = [100, 200]
clus_methods = ["CompassK", "CompassBikmeans", "CompassPca"]
clus_search_methods = ["CompassBikmeans", "CompassBikmeansCg", "CompassBikmeansIcg"]


# Compare clustering methods.
# Choose one clustering method according to these figures,
# potentially one for each dataset.
def compare_clustering_methods():
  for da in DA_S:
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=clus_methods,
      anno="CoC",
      d_m_b=d_m_b_M32,
      nrel_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-clus",
    )
    draw_qps_comp_fixing_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=clus_methods,
      anno="CoC",
      d_m_b=d_m_b_M32,
      nrel_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-clus",
    )


# Compare cluster search methods.
def compare_cluster_search_methods():
  for da in DA_S:
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=clus_search_methods,
      anno="SoS",
      d_m_b=d_m_b_M32,
      nrel_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-clus-search",
    )
    draw_qps_comp_fixing_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=clus_search_methods,
      anno="SoS",
      d_m_b=d_m_b_M32,
      nrel_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-clus-search",
    )


# Compare with iRangeGraph and SeRF.
def compare_with_sotas():
  for da in DA_S:
    draw_qps_comp_wrt_recall_by_dataset_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=d_m_b_M32,
      nrel_s=nrel_100,
      prefix=f"cherrypick{da}d-10",
    )
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=d_m_b_M32,
      nrel_s=nrel_100,
      prefix=f"cherrypick{da}d-10",
    )
    draw_qps_comp_fixing_recall_by_dataset_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=d_m_b_M32,
      nrel_s=nrel_100,
      prefix=f"cherrypick{da}d-10",
    )
    draw_qps_comp_fixing_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=d_m_b_M32,
      nrel_s=nrel_100,
      prefix=f"cherrypick{da}d-10",
    )


def compare_varying_nrel():
  for da in DA_S:
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=clus_search_methods,
      anno="CoC",
      d_m_b=d_m_b_M32,
      nrel_s=nrel_100_200,
      prefix=f"cherrypick{da}d-10/varying-nrel",
    )


def compare_varying_M():
  d_m_b_M32_64 = {d: {} for d in DATASETS}
  for d in ("sift", "audio"):
    for m in clus_search_methods:
      d_m_b_M32_64[d][m] = ["M_16_efc_200_nlist_10000", "M_32_efc_200_nlist_10000"]
  for d in ("gist", "video", "crawl", "glove100"):
    for m in clus_search_methods:
      d_m_b_M32_64[d][m] = ["M_16_efc_200_nlist_20000", "M_32_efc_200_nlist_20000"]

  for da in DA_S:
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=clus_search_methods,
      anno="SoS",
      d_m_b=d_m_b_M32_64,
      nrel_s=nrel_100_200,
      prefix=f"cherrypick{da}d-10/varying-M",
    )


def compare_varying_nlist():
  d_m_b_M32_smallnlist = {d: {} for d in DATASETS}
  for d in ("sift", "audio"):
    for m in clus_search_methods:
      d_m_b_M32_smallnlist[d][m] = [f"M_16_efc_200_nlist_{nlist}" for nlist in [5000, 10000]]
  for d in ("gist", "video", "crawl", "glove100"):
    for m in clus_search_methods:
      d_m_b_M32_smallnlist[d][m] = [f"M_16_efc_200_nlist_{nlist}" for nlist in [10000, 20000]]

  for da in DA_S:
    draw_qps_comp_wrt_recall_by_dataset_selectivity(
      da=da,
      datasets=DATASETS,
      methods=clus_search_methods,
      anno="nlist",
      d_m_b=d_m_b_M32_smallnlist,
      nrel_s=nrel_100,
      prefix=f"cherrypick{da}d-10/varying-nlist",
    )


if __name__ == "__main__":
  compare_clustering_methods()
  compare_cluster_search_methods()
  compare_with_sotas()
  compare_varying_nrel()
  compare_varying_M()
  compare_varying_nlist()
