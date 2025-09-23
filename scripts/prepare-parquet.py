# I need to read a file. This files is in binary format and consists of many rows.
# Each row begins with a uint32_t indicating the the number of float32 numbers followed,
# which represents a vector.
# After reading in those vectors, I need to store them in a parquet file.
# The parquet file should have two columns: id and embedding.
# Ths id would just be the index of the vector in the file.

import pandas as pd
import numpy as np

base_template = "/home/chunxy/repos/Compass/data/{}_base.float32"
attr_template = "/home/chunxy/repos/Compass/data/attr/{}_1_10000.value.bin"
out_template = "/home/chunxy/repos/Compass/data/{}.parquet"

datasets = ["sift", "crawl", "glove100", "video", "audio"]
dims = {"sift": 128, "crawl": 300, "glove100": 100, "video": 1024, "audio": 128}

for d in datasets:
  base_file = base_template.format(d)
  attr_file = attr_template.format(d)
  with open(attr_file, "rb") as f:
    raw = np.fromfile(f, dtype=np.float32).reshape((-1, 2)) # 1 for number of attributes, 1 for the attribute
  attrs = raw[:, 1]

  dim = dims[d]
  with open(base_file, "rb") as f:
    # I want to store the embeddings into a Parquet file together with the id.
    embeddings = np.fromfile(f, dtype=np.float32).reshape((-1, dim))
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
      "attr": attrs,
      "embedding": embedding_rows,
    })

    out_file = out_template.format(d)
    df.to_parquet(out_file, engine="pyarrow", index=False)
