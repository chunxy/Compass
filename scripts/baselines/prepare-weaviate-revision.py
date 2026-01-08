import weaviate
from weaviate.classes.config import Configure, VectorDistances
from utils import REVISION_CARDS, DATASET_M
from utils import load_fvecs, load_float32
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
    if name not in REVISION_CARDS.keys():
      raise ValueError(f"Invalid dataset name: {name}. Must be one of {list(REVISION_CARDS.keys())}")

  build_time = {}
  # Step 1.1: Connect to your local Weaviate instance
  with weaviate.connect_to_local() as client:
    for d in args.names:
      # Step 1.2: Create a collection
      database_name = f"{d}_revision".replace("-", "_")
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

      card0 = REVISION_CARDS[d][0]
      vectors = load_fvecs(card0.base_path, card0.n_base, card0.dim)
      # Step 1.3: Import objects
      data_objects = [
        {
          "properties": {
            "vid": i,
          },
          "vector": vector.tolist(),
        } for i, vector in enumerate(vectors)
      ]

      # Load attribute values.
      for card in REVISION_CARDS[d]:
        if card.wtype in ["skewed", "correlated", "real"]:
          attrs = load_float32(card.attr_path, card.n_base, card.attr_dim)
          if (vectors.shape[0] != attrs.shape[0]):
            raise ValueError(f"Vectors and attrs have different lengths for {d}_{card.wtype}")
          if (attrs.shape[1] != card.attr_dim):
            raise ValueError(f"Loaded attributes have different dimension for {d}_{card.wtype}")
          if (vectors.shape[1] != card.dim):
            raise ValueError(f"Loaded vectors have different dimensions for {d}_{card.wtype}")
          for i, attr in enumerate(attrs):
            if card.attr_dim == 1:
              data_objects[i]["properties"][f"{card.wtype}"] = float(attr[0])
            else:
              for j in range(card.attr_dim):
                data_objects[i]["properties"][f"{card.wtype}{j+1}"] = float(attr[j])

      # Insert the objects with vectors
      db = client.collections.get(database_name)
      time_start = time.perf_counter_ns()
      bs = 100
      with db.batch.fixed_size(batch_size=bs) as batch:
        for obj in data_objects:
          batch.add_object(properties=obj["properties"], vector=obj["vector"])
      time_end = time.perf_counter_ns()
      time_taken = time_end - time_start
      build_time[f"{d}_revision"] = time_taken / 1e9
      print(f"Imported {len(data_objects)} objects into the {database_name}")

    with open(BUILD_DIR / "build_time_in_seconds_revision.json", "w") as f:
      json.dump(build_time, f, indent=2)
