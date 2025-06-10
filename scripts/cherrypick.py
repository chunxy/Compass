from config import COMPASS_METHODS, DA_S, DATASETS, METHODS
from summarize import (
  draw_qps_comp_fixed_recall_by_dataset_selectivity,
  draw_qps_comp_fixed_recall_by_selectivity,
  draw_qps_comp_wrt_recall_by_dataset_selectivity,
  draw_qps_comp_wrt_recall_by_selectivity,
)

if __name__ == "__main__":
  compass_nlist_10000 = {m: ["M_16_efc_200_nlist_10000"] for m in COMPASS_METHODS}
  compass_nlist_20000 = {m: ["M_16_efc_200_nlist_20000"] for m in COMPASS_METHODS}
  d_m_b = {
    **{
      d: compass_nlist_10000
      for d in ("sift", "audio")
    },
    **{
      d: compass_nlist_20000
      for d in ("gist", "video", "crawl", "glove100")
    },
  }
  for d in DATASETS:
    d_m_b[d]["iRangeGraph"] = ["M_32_efc_200"]
    d_m_b[d]["SeRF"] = ["M_32_efc_200_efmax_500"]

  nrel_s = [100]

  # Compare clustering methods.
  # Choose one clustering method according to these figures,
  # potentially one for each dataset.
  clus_methods = ["CompassK", "CompassBikmeans", "CompassPca"]
  for da in DA_S:
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=clus_methods,
      anno="CoC",
      d_m_b=d_m_b,
      nrel_s=nrel_s,
      prefix="cherrypick",
    )
    draw_qps_comp_fixed_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=clus_methods,
      anno="CoC",
      d_m_b=d_m_b,
      nrel_s=nrel_s,
      prefix="cherrypick",
    )

  # Compare clustering search methods.
  clus_search_methods = ["CompassBikmeans", "CompassBikmeansCg"]
  for da in DA_S:
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=clus_search_methods,
      anno="SoS",
      d_m_b=d_m_b,
      nrel_s=nrel_s,
      prefix="cherrypick",
    )
    draw_qps_comp_fixed_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=clus_search_methods,
      anno="SoS",
      d_m_b=d_m_b,
      nrel_s=nrel_s,
      prefix="cherrypick",
    )

  # Compare with iRangeGraph and SeRF.
  for da in DA_S:
    draw_qps_comp_wrt_recall_by_dataset_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=d_m_b,
      nrel_s=nrel_s,
      prefix="cherrypick",
    )
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=d_m_b,
      nrel_s=nrel_s,
      prefix="cherrypick",
    )
    draw_qps_comp_fixed_recall_by_dataset_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=d_m_b,
      nrel_s=nrel_s,
      prefix="cherrypick",
    )
    draw_qps_comp_fixed_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=d_m_b,
      nrel_s=nrel_s,
      prefix="cherrypick",
    )

