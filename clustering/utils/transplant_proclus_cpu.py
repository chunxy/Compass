from pathlib import Path
import shutil
dataset_config = {
  "sift": [(1000, 64)] + [(10000, 64)],
  "audio": [(1000, 64)] + [(10000, 64)],
  "glove100": [(1000, 64)] + [(1000, 50)],
  "crawl": [(2000, 64)] + [(2000, 100)],
  "gist": [(1000, 128)] + [(1000, 320)],
  "video": [(10000, 128)] + [(1000, 128)],
}

template_medoids_path = "/home/chunxy/repos/Compass/data/proclus-cpu/{}.{}.{}.proclus.medoids"
template_subspaces_path = "/home/chunxy/repos/Compass/data/proclus-cpu/{}.{}.{}.proclus.subspaces"
template_new_medoids_path = "/home/chunxy/repos/Compass/checkpoints/Proclus-CPU/{}/{}-{}.medoids"
template_new_subspaces_path = "/home/chunxy/repos/Compass/checkpoints/Proclus-CPU/{}/{}-{}.subspaces"

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
