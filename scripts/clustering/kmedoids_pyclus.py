from pyclustering.cluster.kmedoids import kmedoids
import numpy as np

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
parser.add_argument("--nlist", type=int, required=True, help="Number of clusters")
args = parser.parse_args()

if args.name not in datasets:
  raise ValueError(f"Invalid dataset name: {args.name}. Must be one of {list(datasets.keys())}")
name = args.name
d = datasets[name]
nlist = args.nlist


# Example data (replace this with your actual data)
file = f"/home/chunxy/repos/Compass/data/{name}_base.float32"
training_data = np.fromfile(file, dtype=np.float32).reshape((-1, d)).tolist()

# Set random initial medoids.
ordinals = np.arange(len(training_data))
np.random.shuffle(ordinals)
initial_medoids = ordinals[:nlist].tolist()

# Create instance of K-Medoids algorithm.
kmedoids_instance = kmedoids(training_data, initial_medoids)

# Run cluster analysis and obtain results.
kmedoids_instance.process()
medoids_inds = kmedoids_instance.get_medoids()
medoids = training_data[medoids_inds]
medoids.tofile(f"/home/chunxy/repos/Compass/data/{name}.{nlist}.kmedoids.pyclus.medoids")