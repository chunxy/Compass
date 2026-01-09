from pymilvus import MilvusClient
from utils import DA_S, CARDS, DATASET_M, N_QUERIES, TOPK, EFS_S
from utils import load_fvecs, load_ivecs
import time
import json
from pathlib import Path
import argparse

LOGS_DIR = Path("/home/chunxy/milvus/logs_10")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Bench Milvus")
  parser.add_argument("--names", type=str, nargs='+', required=False, help="Dataset name to process")
  args = parser.parse_args()
  # args.names = ["sift-dedup"]

  for name in args.names:
    if name not in CARDS.keys():
      raise ValueError(f"Invalid dataset name: {name}. Must be one of {list(CARDS.keys())}")

  # Token form
  client = MilvusClient(
    uri="http://127.0.0.1:19530",
    token="root:Milvus",
  )

  for d in args.names:
    for da in DA_S:
      database_name = f"{d}_{da}".replace("-", "_")

      for card in CARDS[d][da]:
        query_vectors_np = load_fvecs(card.query_path, card.n_queries, card.dim)[:N_QUERIES]
        query_vectors = [query_vector.tolist() for query_vector in query_vectors_np]
        predicate = " and ".join([f"attr_{i} >= {card.interval[0][i]} and attr_{i} <= {card.interval[1][i]}" for i in range(card.attr_dim)])
        for efs in EFS_S:
          time_start = time.perf_counter_ns()
          res = []
          bs = 1
          for i in range(0, len(query_vectors), bs):
            st, ed = i, min(i + bs, len(query_vectors))
            partial_res = client.search(
              collection_name=database_name,
              anns_field="vector",
              data=query_vectors[st:ed],
              limit=TOPK,
              filter=predicate,
              output_fields=["id"],
              search_params={
                # "hints": "iterative_filter",
                "params": {
                  "ef": efs
                },
              }
            )
            res.extend(partial_res)
          time_end = time.perf_counter_ns()
          time_taken = time_end - time_start

          groundtruth = load_ivecs(card.groundtruth_path, card.n_queries, card.n_groundtruth)[:N_QUERIES, :TOPK]
          recall = 0
          for i, hits in enumerate(res):
            for hit in hits:
              if hit["id"] in groundtruth[i]:
                recall += 1

          logs_dir = LOGS_DIR / card.workload / f"M_{DATASET_M[d]}_efc_{200}" / f"efs_{efs}"
          logs_dir.mkdir(parents=True, exist_ok=True)
          json_path = logs_dir / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}.json"
          print(f"Saving to {json_path}")
          with open(json_path, "w") as f:
            json.dump({"aggregated": {
              "recall": recall / N_QUERIES / TOPK,
              "qps": N_QUERIES * 1e9 / time_taken,
            }}, f)
