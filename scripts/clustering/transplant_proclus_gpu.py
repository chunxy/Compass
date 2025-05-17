from pathlib import Path
import shutil
dataset_config = {
  "sift": [(500, 32), (500, 64), (1000, 32), (1000, 64)],
  "glove100": [(500, 32), (500, 64), (1000, 32), (1000, 64)],
  "gist": [(500, 32), (500, 64), (1000, 32), (1000, 64)],
  "crawl": [(500, 32), (500, 64), (1000, 32), (1000, 64)],
  "video": [(500, 32), (500, 64), (1000, 32), (1000, 64)],
  "audio": [(500, 32), (500, 64), (1000, 32), (1000, 64)],
}

template_medoids_path = "/home/chunxy/repos/Compass/data/proclus-gpu/{}.{}.{}.proclus.medoids"
template_subspaces_path = "/home/chunxy/repos/Compass/data/proclus-gpu/{}.{}.{}.proclus.subspaces"
template_new_medoids_path = "/home/chunxy/repos/Compass/checkpoints/Proclus/{}/{}-{}.medoids"
template_new_subspaces_path = "/home/chunxy/repos/Compass/checkpoints/Proclus/{}/{}-{}.subspaces"

for d, configs in dataset_config.items():
  for config in configs:
    nlist, dim = config
    medoids_path = template_medoids_path.format(d, nlist, dim)
    subspaces_path = template_subspaces_path.format(d, nlist, dim)
    if not Path(medoids_path).exists() or not Path(subspaces_path).exists(): continue
    # copy medoids and subspaces to a new file
    new_medoids_path = template_new_medoids_path.format(d, nlist, dim)
    new_subspaces_path = template_new_subspaces_path.format(d, nlist, dim)
    shutil.copy(medoids_path, new_medoids_path)
    shutil.copy(subspaces_path, new_subspaces_path)
