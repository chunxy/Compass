from config import DATASETS, METHODS
from scripts.summarize import (
  draw_qps_comp_fixed_recall_by_dataset_selectivity,
  draw_qps_comp_fixed_recall_by_selectivity,
  draw_qps_comp_wrt_recall_by_dataset_selectivity,
  draw_qps_comp_wrt_recall_by_selectivity,
)

if __name__ == "__main__":
  nlist_10000 = {m: ["M_16_efc_200_nlist_10000"] for m in METHODS}
  nlist_20000 = {m: ["M_16_efc_200_nlist_20000"] for m in METHODS}
  d_m_b = {
    **{
      d: nlist_10000
      for d in ("sift", "audio")
    },
    **{
      d: nlist_20000
      for d in ("gist", "video", "crawl", "glove100")
    },
  }
  for da in (1, 2, 3, 4):
    draw_qps_comp_wrt_recall_by_dataset_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      d_m_b=d_m_b,
      prefix="cherrypick",
    )
    draw_qps_comp_wrt_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      d_m_b=d_m_b,
      prefix="cherrypick",
    )
    draw_qps_comp_fixed_recall_by_dataset_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=d_m_b,
      prefix="cherrypick",
    )
    draw_qps_comp_fixed_recall_by_selectivity(
      da=da,
      datasets=DATASETS,
      methods=METHODS,
      anno="MoM",
      d_m_b=d_m_b,
      prefix="cherrypick",
    )
