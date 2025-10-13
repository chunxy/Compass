import re
from pathlib import Path

from config import (
  COMPASS_METHODS,
  BASE_METHODS,
  DA_S,
  M_DA_RUN,
  M_GROUP_DATASET,
  M_PARAM,
  D_ARGS,
  M_ARGS,
)

EXP_ROOT = Path("/home/chunxy/repos/Compass/runs/exp")


def __enclose_for(ele_name, collection_name, body):
  return \
f'''for {ele_name} in ${{{collection_name}[@]}}; do
{body}
done'''


def compose():
  for da in DA_S:
    for m in COMPASS_METHODS + BASE_METHODS:
      if m == "ACORN":
        continue
      if da not in M_DA_RUN[m]:
        continue
      idx = 0
      for group, datasets in M_GROUP_DATASET[m].items():
        parts = [p.lower() for p in re.findall(r'[A-Z][a-z]*', m)]
        for d in datasets:
          for M in D_ARGS[d].get("M", M_ARGS[m].get("M", [1])):
            idx += 1
            filename = f"{da}d-" + "-".join(parts[1:]) + f"-exp-{idx}.sh"
            with open(EXP_ROOT / filename, "w") as f:
              f.write(f'dataset={d}\n')
              for bp in M_PARAM[m]["build"]:
                if bp == "M":
                  f.write(f'{bp}_s=({M})\n')
                else:
                  f.write(f'{bp}_s=({" ".join(map(str, D_ARGS[d].get(bp, M_ARGS[m][bp])))})\n')
              for sp in M_PARAM[m]["search"]:
                f.write(f'{sp}_s=({" ".join(map(str, D_ARGS[d].get(sp, M_ARGS[m][sp])))})\n')

              build_string = " ".join(map(lambda x: f"--{x} ${{{x}}}", M_PARAM[m]["build"]))
              search_string = " ".join(map(lambda x: f"--{x} ${{{x}_s[@]}}", M_PARAM[m]["search"]))
              intervals = M_DA_RUN[m][da]
              inner_tmpl = \
'''/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-{} \
--datacard ${{dataset}}_{}_10000_float32 \
--l {} --r {} --k 10 {} {}'''
              inner = '\n'.join([
                inner_tmpl.format(
                  ("1d-" if da == 1 else "") + "-".join(parts[1:]),
                  da,
                  " ".join(map(str, itvl[0])),
                  " ".join(map(str, itvl[1])),
                  build_string,
                  search_string,
                ) for itvl in intervals
              ])

              for bp in M_PARAM[m]["build"][::-1]:
                inner = __enclose_for(bp, f"{bp}_s", inner)
              f.write(inner)
              f.write('\n')


if __name__ == "__main__":
  compose()
