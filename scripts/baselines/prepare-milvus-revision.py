from pymilvus import MilvusClient, DataType
from utils import REVISION_CARDS, DATASET_M
from utils import load_fvecs, load_float32
import argparse
from pathlib import Path
import time
import json

BUILD_DIR = Path("/home/chunxy/milvus")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Prepare Milvus")
  parser.add_argument("--names", type=str, nargs='+', required=False, help="Dataset name to process")
  args = parser.parse_args()
  args.names = ["sift-dedup"]

  for name in args.names:
    if name not in REVISION_CARDS.keys():
      raise ValueError(f"Invalid dataset name: {name}. Must be one of {list(REVISION_CARDS.keys())}")

  # Token form
  client = MilvusClient(
    uri="http://127.0.0.1:19530",
    token="root:Milvus",
  )

  build_time = {}
  for d in args.names:
    database_name = f"{d}_revision".replace("-", "_")
    card0 = REVISION_CARDS[d][0]

    # 3.1. Create schema
    client.drop_collection(collection_name=database_name)
    schema = MilvusClient.create_schema(
      auto_id=False,
      enable_dynamic_field=False,
    )

    # 3.2. Add fields to schema
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=card0.dim)
    for card in REVISION_CARDS[d]:
      if card.wtype not in ["skewed", "correlated", "real"]:
        continue
      if card.attr_dim == 1:
        schema.add_field(field_name=f"{card.wtype}", datatype=DataType.FLOAT, dim=1)
      else:
        for i in range(1, card.attr_dim + 1):
          schema.add_field(field_name=f"{card.wtype}{i}", datatype=DataType.FLOAT, dim=1)

    # 3.3. Prepare index parameters
    index_params = client.prepare_index_params()

    # 3.4. Add indexes
    # index_params.add_index(field_name="id", index_type="")
    # index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="L2")
    index_params.add_index(
      field_name="vector",  # Name of the vector field to be indexed
      index_type="HNSW",  # Type of the index to create
      index_name="vector_index",  # Name of the index to create
      metric_type="L2",  # Metric type used to measure similarity
      params={
        "M": DATASET_M[d],  # Maximum number of neighbors each node can connect to in the graph
        "efConstruction": 200  # Number of candidate neighbors considered for connection during index construction
      }  # Index building params
    )

    # 3.5. Create a collection with the index loaded simultaneously
    client.create_collection(collection_name=database_name, schema=schema, index_params=index_params)
    client.get_load_state(collection_name=database_name)

    vectors = load_fvecs(card0.base_path, card0.n_base, card0.dim)
    data = [{"id": i, "vector": vector.tolist()} for i, vector in enumerate(vectors)]
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
            data[i][f"{card.wtype}"] = float(attr[0])
          else:
            for j in range(card.attr_dim):
              data[i][f"{card.wtype}{j+1}"] = float(attr[j])

    time_start = time.perf_counter_ns()
    bs = 1000
    for i in range(0, len(data), bs):
      st, ed = i, min(i + bs, len(data))
      res = client.insert(collection_name=database_name, data=data[st:ed])
    time_end = time.perf_counter_ns()
    time_taken = time_end - time_start
    build_time[f"{d}_revision"] = time_taken / 1e9
    print(f"Imported {len(data)} objects into the {database_name}")

  with open(BUILD_DIR / "build_time_in_seconds_revision.json", "w") as f:
    json.dump(build_time, f, indent=2)
