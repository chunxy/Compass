from sklearn.manifold import LocallyLinearEmbedding as LLE
import numpy as np
import pickle
import argparse

datasets = {
  "gist": 960,
  "crawl": 300,
  "glove100": 100,
  "audio": 128,
  "video": 1024,
  "sift": 128,
}


parser = argparse.ArgumentParser(description="Clustering script")
parser.add_argument("--name", type=str, required=True, help="Dataset name to process")
parser.add_argument("--n_neighbors", type=int, required=True, help="Number of neighbors")
parser.add_argument("--n_components", type=int, required=True, help="Number of components")
parser.add_argument("--n_samples", type=int, required=True, help="Number of samples")
args = parser.parse_args()

if args.name not in datasets:
  raise ValueError(f"Invalid dataset name: {args.name}. Must be one of {list(datasets.keys())}")

name = args.name
d = datasets[name]
n_neighbors = args.n_neighbors
n_components = args.n_components
n_samples = args.n_samples
base_file = f"/home/chunxy/repos/Compass/data/{name}_base.float32"
base_data = np.fromfile(base_file, dtype=np.float32).reshape((-1, d))
sample_data = np.random.permutation(base_data)[:n_samples]
query_data = np.fromfile(f"/home/chunxy/repos/Compass/data/{name}_query.float32", dtype=np.float32).reshape((-1, d))

# Perform BisectingKMeans clustering
lle = LLE(n_neighbors=n_neighbors, n_components=n_components)
lle.fit(sample_data)

base_x = lle.transform(base_data).astype(np.float32)
query_x = lle.transform(query_data).astype(np.float32)
base_x.tofile(f"/home/chunxy/repos/Compass/data/llem/{name}.{n_neighbors}.{n_components}.base.float32")
query_x.tofile(f"/home/chunxy/repos/Compass/data/llem/{name}.{n_neighbors}.{n_components}.query.float32")

with open(f"/home/chunxy/repos/Compass/data/llem/{name}.{n_neighbors}.{n_components}.lle.pkl", "wb") as f:
  pickle.dump(lle, f)

