# I need to read a file. This files is in binary format and consists of many rows.
# Each row begins with a uint32_t indicating the the number of float32 numbers followed,
# which represents a vector.
# After reading in those vectors, I need to store them in a parquet file.
# The parquet file should have two columns: id and embedding.
# Ths id would just be the index of the vector in the file.

import pandas as pd
import numpy as np
from collections import namedtuple

attr_template = "/home/chunxy/repos/Compass/data/attr/{}_{}_{}.{}.value.bin"
attribute = namedtuple("attribute", ["da", "span", "type"])
attributes = [
  attribute(1, 30, "skewed"),
  attribute(2, 20, "correlated"),
  attribute(2, 20, "anticorrelated"),
  # attribute(1, 30, "real"), # to be added
]
out_template = "/home/chunxy/repos/Compass/data/navix/{}_revision.value.parquet"

datasets = ["audio-dedup", "sift-dedup", "crawl", "glove100", "video-dedup", "gist-dedup"]
dataset_size = {
  "sift-dedup": 1000000 - 14538,
  "audio-dedup": 1000000,
  "gist-dedup": 1000000 - 17306,
  "video-dedup": 1000000,
  "glove100": 1183514,
  "crawl": 1989995,
}
# dims = {"sift-dedup": 128, "crawl": 300, "glove100": 100, "video-dedup": 1024, "audio-dedup": 128, "gist-dedup": 960}

for d in datasets:
  ids = np.arange(dataset_size[d], dtype=np.int64)
  df = pd.DataFrame({
    "id": ids,
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
  df.to_parquet(out_template.format(d), engine="pyarrow", index=False)
