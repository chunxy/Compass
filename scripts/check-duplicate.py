# I need to read a file. This files is in binary format and consists of many rows.
# Each row begins with a uint32_t indicating the the number of float32 numbers followed,
# which represents a vector.
# After reading in those vectors, I need to store them in a parquet file.
# The parquet file should have two columns: id and embedding.
# Ths id would just be the index of the vector in the file.

import numpy as np

base_template = "/home/chunxy/repos/Compass/data/{}_base.float32"
attr_template = "/home/chunxy/repos/Compass/data/attr/{}_1_10000.value.bin"
out_template = "/home/chunxy/repos/Compass/data/{}_base.parquet"

datasets = ["sift", "crawl", "glove100", "video", "audio", "gist"]
dims = {"sift": 128, "crawl": 300, "glove100": 100, "video": 1024, "audio": 128, "gist": 960}

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

    # Count exact duplicate vectors efficiently by viewing each row as a single
    # void blob of bytes and using numpy.unique on that representation.
    row_size_bytes = embeddings.strides[0]
    row_view = embeddings.view(np.dtype((np.void, row_size_bytes))).ravel()
    _, counts = np.unique(row_view, return_counts=True)
    duplicate_rows = int((counts - 1)[counts > 1].sum())
    duplicate_groups = int((counts > 1).sum())
    print(f"{d}: vectors={embeddings.shape[0]}, dim={dim}, duplicate_rows={duplicate_rows}, duplicate_groups={duplicate_groups}")
