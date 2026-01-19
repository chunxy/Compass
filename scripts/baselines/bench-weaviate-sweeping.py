import weaviate
import json
from utils import CARDS, DA_S, N_QUERIES, TOPK, DATASET_M, EFS_S
from utils import load_fvecs, load_ivecs
from weaviate.classes.query import Filter
from weaviate.classes.config import Reconfigure, VectorFilterStrategy
import time
from pathlib import Path
import argparse

LOGS_DIR = Path("/home/chunxy/weaviate/logs_10")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Bench Weaviate")
  parser.add_argument("--names", type=str, nargs='+', required=False, help="Dataset name to process")
  args = parser.parse_args()
  # args.names = ["sift-dedup"]

  for name in args.names:
    if name not in CARDS.keys():
      raise ValueError(f"Invalid dataset name: {name}. Must be one of {list(CARDS.keys())}")

  # Step 2.1: Connect to your local Weaviate instance
  with weaviate.connect_to_local() as client:
    for d in args.names:
      for da in DA_S:
        # Step 1.2: Create a collection
        database_name = f"{d}_{da}".replace("-", "_").capitalize()
        # Step 2.2: Use this collection
        db = client.collections.use(database_name)

        # Step 2.3: Perform a vector search with NearVector
        for card in CARDS[d][da]:
          for efs in EFS_S:
            db.config.update(
              vector_config=Reconfigure.Vectors.update(
                name="vector",
                vector_index_config=Reconfigure.VectorIndex.hnsw(
                  ef=efs,
                  filter_strategy=VectorFilterStrategy.SWEEPING,
                ),
              )
            )

            query_vectors = load_fvecs(card.query_path, card.n_queries, card.dim)[:N_QUERIES]
            predicate = Filter.all_of([Filter.by_property(f"attr_{i}").greater_or_equal(card.interval[0][i]) for i in range(card.attr_dim)] +
                                      [Filter.by_property(f"attr_{i}").less_or_equal(card.interval[1][i]) for i in range(card.attr_dim)])

            responses = [0 for i in range(N_QUERIES)]
            time_start = time.perf_counter_ns()
            for i, query_vector in enumerate(query_vectors):
              # Search for top-10.
              responses[i] = db.query.near_vector(
                near_vector=query_vector,
                filters=predicate,
                limit=TOPK,
                return_properties="vid",
              )
            time_end = time.perf_counter_ns()
            time_taken = time_end - time_start

            groundtruth = load_ivecs(card.groundtruth_path, card.n_queries, card.n_groundtruth)[:N_QUERIES, :TOPK]
            recall = 0
            for i, response in enumerate(responses):
              for obj in response.objects:
                if obj.properties["vid"] in groundtruth[i]:
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
