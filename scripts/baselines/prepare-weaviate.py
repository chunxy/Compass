import weaviate
from weaviate.classes.config import Configure, VectorDistances
from utils import DA_S, CARDS, DATASET_M
from utils import load_fvecs, load_attr
import time
import json
from pathlib import Path
import argparse

BUILD_DIR = Path("/home/chunxy/weaviate")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Prepare Weaviate")
  parser.add_argument("--names", type=str, nargs='+', required=False, help="Dataset name to process")
  args = parser.parse_args()
  # args.names = ["sift-dedup"]

  for name in args.names:
    if name not in CARDS.keys():
      raise ValueError(f"Invalid dataset name: {name}. Must be one of {list(CARDS.keys())}")

  build_time = {}
  # Step 1.1: Connect to your local Weaviate instance
  with weaviate.connect_to_local() as client:
    for da in DA_S:
      for d in args.names:
        # Step 1.2: Create a collection
        database_name = f"{d}_{da}".replace("-", "_")
        client.collections.delete(database_name)
        schema = client.collections.create(
          name=database_name,
          vector_config=Configure.Vectors.self_provided(
            name="vector",
            vector_index_config=Configure.VectorIndex.hnsw(
              max_connections=DATASET_M[d],
              ef_construction=200,
              distance_metric=VectorDistances.L2_SQUARED,
            )
          )  # No automatic vectorization since we're providing vectors
        )

        card = CARDS[d][da][0]
        vectors = load_fvecs(card.base_path, card.n_base, card.dim)
        attrs = load_attr(card.attr_path, card.n_base, card.attr_dim)

        if (vectors.shape[0] != attrs.shape[0]):
          raise ValueError(f"Vectors and attrs have different lengths for {d}_{da}")
        if (attrs.shape[1] != card.attr_dim):
          raise ValueError(f"Loaded attributes have different dimension for {d}_{da}")
        if (vectors.shape[1] != card.dim):
          raise ValueError(f"Loaded vectors have different dimension for {d}_{da}")

        # Step 1.3: Import objects
        data_objects = [{
          "properties": {
            **{
              f"attr_{i}": float(attr[i])
              for i in range(card.attr_dim)
            },
            "vid": i,
          },
          "vector": vector.tolist(),
        }
                        for i, (attr, vector) in enumerate(zip(attrs, vectors))]

        # Insert the objects with vectors
        db = client.collections.get(database_name)
        time_start = time.perf_counter_ns()
        bs = 100
        with db.batch.fixed_size(batch_size=bs) as batch:
          for obj in data_objects:
            batch.add_object(properties=obj["properties"], vector=obj["vector"])
        time_end = time.perf_counter_ns()
        time_taken = time_end - time_start
        build_time[f"{d}_{da}"] = time_taken / 1e9
        print(f"Imported {len(data_objects)} objects into the {database_name}")

      with open(BUILD_DIR / "build_time_in_seconds.json", "w") as f:
        json.dump(build_time, f, indent=2)
