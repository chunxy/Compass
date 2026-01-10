# I need to read a file. This files is in binary format and consists of many rows.
# Each row begins with a uint32_t indicating the the number of float32 numbers followed,
# which represents a vector.
# After reading in those vectors, I need to store them in a parquet file.
# The parquet file should have two columns: id and embedding.
# Ths id would just be the index of the vector in the file.

import pandas as pd
import numpy as np
from collections import namedtuple

base_template = "/opt/nfs_dcc/chunxy/datasets/{}/{}_base.fvecs"
attr_template = "/home/chunxy/repos/Compass/data/attr/{}_{}_{}.{}.value.bin"
out_template = "/home/chunxy/repos/Compass/data/{}_base.parquet"

datasets = ["flickr", "deep10m"]
dims = {"flickr": 512, "deep10m": 96}

attribute = namedtuple("attribute", ["da", "span", "type"])
attributes = [
  attribute(1, 30, "skewed"),
  attribute(2, 20, "correlated"),
  attribute(2, 20, "anticorrelated"),
]

for d in datasets:
  base_file = base_template.format(d, d)

  dim = dims[d]
  with open(base_file, "rb") as f:
    embeddings = np.fromfile(f, dtype=np.float32).reshape((-1, 1 + dim))[:, 1:]
    ids = np.arange(embeddings.shape[0], dtype=np.int64)
    # Validate shapes
    if embeddings.shape[0] != ids.shape[0]:
      raise ValueError(f"Mismatch in number of embeddings and ids for dataset {d}: {embeddings.shape[0]} vs {ids.shape[0]}")
    if embeddings.shape[1] != dim:
      raise ValueError(f"Embedding dimension mismatch for dataset {d}: expected {dim}, got {embeddings.shape[1]}")
    # Ensure each embedding remains float32 for Parquet list<float32>
    embedding_rows = [embeddings[i].copy() for i in range(embeddings.shape[0])]
    df = pd.DataFrame({
      "id": ids,
      "embedding": embedding_rows,
    })

    for w in attributes:
      attr_file = attr_template.format(d, w.da, w.span, w.type)
      with open(attr_file, "rb") as f:
        raw = np.fromfile(f, dtype=np.float32).reshape((-1, w.da))
      if w.da == 1:
        df[f"{w.type}"] = raw.flatten()
      else:
        for i in range(w.da):
          df[f"{w.type}{i + 1}"] = raw[:, i]

    out_file = out_template.format(d)
    df.to_parquet(out_file, engine="pyarrow", index=False)
