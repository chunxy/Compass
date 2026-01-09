from pymilvus import MilvusClient
from utils import REVISION_CARDS, DATASET_M, N_QUERIES, TOPK, EFS_S
from utils import load_fvecs, load_ivecs, load_float32
import time
import json
from pathlib import Path
import argparse

LOGS_DIR = Path("/home/chunxy/milvus/logs_10")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Bench Milvus")
  parser.add_argument("--names", type=str, nargs='+', required=False, help="Dataset name to process")
  parser.add_argument("--wtypes", type=str, nargs='+', required=False, help="Workload types to process")
  args = parser.parse_args()
  # args.names = ["sift-dedup"]
  # args.wtypes = ["skewed", "correlated", "anticorrelated"]

  for name in args.names:
    if name not in REVISION_CARDS.keys():
      raise ValueError(f"Invalid dataset name: {name}. Must be one of {list(REVISION_CARDS.keys())}")

  # Token form
  client = MilvusClient(
    uri="http://127.0.0.1:19530",
    token="root:Milvus",
  )

  for d in args.names:
    database_name = f"{d}_revision".replace("-", "_")

    for card in REVISION_CARDS[d]:
      if args.wtypes is not None and card.wtype not in args.wtypes:
        continue
      query_vectors_np = load_fvecs(card.query_path, card.n_queries, card.dim)[:N_QUERIES]
      if card.wtype == "skewed" or card.wtype == "correlated" or card.wtype == "anticorrelated":
        rg = load_float32(card.rg_path, card.n_queries * 2, card.attr_dim)
        l_bounds = rg[:card.n_queries]
        u_bounds = rg[card.n_queries:]
      else:
        rg = load_float32(card.rg_path, card.n_queries, card.attr_dim)
        l_bounds = rg[:card.n_queries]
      query_vectors = [query_vector.tolist() for query_vector in query_vectors_np]
      if card.wtype == "skewed":
        predicates = [f"skewed >= {l_bounds[i][0]} and skewed <= {u_bounds[i][0]}" for i in range(card.n_queries)]
      elif card.wtype == "correlated":
        predicates = [
          f"correlated1 >= {l_bounds[i][0]} and correlated1 <= {u_bounds[i][0]} and correlated2 >= {l_bounds[i][1]} and correlated2 <= {u_bounds[i][1]}"
          for i in range(card.n_queries)
        ]
      elif card.wtype == "anticorrelated":
        predicates = [
          f"anticorrelated1 >= {l_bounds[i][0]} and anticorrelated1 <= {u_bounds[i][0]} and anticorrelated2 >= {l_bounds[i][1]} and anticorrelated2 <= {u_bounds[i][1]}"
          for i in range(card.n_queries)
        ]
      elif card.wtype == "real":
        predicates = [
          f"real1 >= {l_bounds[i][0]} and real1 <= {u_bounds[i][0]} and real2 >= {l_bounds[i][1]} and real2 <= {u_bounds[i][1]}"
          for i in range(card.n_queries)
        ]
      elif card.wtype == "onesided":
        predicates = [f"skewed >= {l_bounds[i][0]}" for i in range(card.n_queries)]
      elif card.wtype == "point":
        predicates = [f"skewed == {l_bounds[i][0]}" for i in range(card.n_queries)]
      elif card.wtype == "negation":
        predicates = [f"skewed != {l_bounds[i][0]}" for i in range(card.n_queries)]
      for efs in EFS_S:
        time_start = time.perf_counter_ns()
        res = []
        for i in range(len(query_vectors)):
          partial_res = client.search(
            collection_name=database_name,
            anns_field="vector",
            data=query_vectors[i:i+1],
            limit=TOPK,
            filter=predicates[i],
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
