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
      if da not in M_DA_RUN[m]:
        continue
      idx = 0
      if m == "ACORN":
        for group, datasets in M_GROUP_DATASET[m].items():
          parts = [p.lower() for p in re.findall(r'[A-Z][a-z]*', m)]
          for d in datasets:
            for M in M_ARGS[m].get("M"):
              idx += 1
              filename = f"{da}d-" + "acorn" + f"-exp-{idx}.sh"
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
                search_string = " ".join(map(lambda x: f"--{x} ${{{x}}}", M_PARAM[m]["search"]))
                nlabels = M_DA_RUN[m][da]
                inner_tmpl = \
  '''/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-acorn \
  --datacard ${{dataset}}_1_{}_int32 --k 10 {} {}'''
                inner = '\n'.join([
                  inner_tmpl.format(
                    nlabel,
                    build_string,
                    search_string,
                  ) for nlabel in nlabels
                ])

                for sp in M_PARAM[m]["search"][::-1]:
                  inner = __enclose_for(sp, f"{sp}_s", inner)
                for bp in M_PARAM[m]["build"][::-1]:
                  inner = __enclose_for(bp, f"{bp}_s", inner)
                f.write(inner)
                f.write('\n')
        continue
      if m == "CompassPostKThCh":
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
  --p {} {} {} --k 10 {} {}'''
                inner = '\n'.join([
                  inner_tmpl.format(
                    "-".join(parts[1:]),
                    da,
                    itvl[0],
                    ("" if da == 1 else "--l ") + " ".join(map(str, range(200, 200 + 100 * (da - 1), 100))),
                    ("" if da == 1 else "--r ") + " ".join(map(str, range(200 + 100 * itvl[0], 200 + 100 * (da - 1) + 100 * itvl[0], 100))),
                    build_string,
                    search_string,
                  ) for itvl in intervals
                ])

                for bp in M_PARAM[m]["build"][::-1]:
                  inner = __enclose_for(bp, f"{bp}_s", inner)
                f.write(inner)
                f.write('\n')
        continue
      k_s = [10]
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
--l {} --r {} --k ${{k}} {} {}'''
              inner = '\n'.join([
                inner_tmpl.format(
                  ("1d-" if (da == 1 and not m.startswith("Compass")) else "") + "-".join(parts[1:]),
                  da,
                  " ".join(map(str, itvl[0])),
                  " ".join(map(str, itvl[1])),
                  build_string,
                  search_string,
                ) for itvl in intervals
              ])

              for bp in M_PARAM[m]["build"][::-1]:
                inner = __enclose_for(bp, f"{bp}_s", inner)
              # if m == "CompassPostKTh" and da == 1:
              #   k_s = [5, 10, 15, 20, 25]
              inner = __enclose_for("k", "k_s", inner)
              inner = (f'k_s=({" ".join(map(str, k_s))})\n') + inner
              f.write(inner)
              f.write('\n')

def compose_revision():
  for da in DA_S:
    for m in ["ACORN"]:
      if da not in M_DA_RUN[m]:
        continue
      idx = 0
      for group, datasets in M_GROUP_DATASET[m].items():
        for d in datasets:
          for M in M_ARGS[m].get("M"):
            idx += 1
            filename = f"{da}d-" + "acorn" + f"-exp-{idx}.sh"
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
'''/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-acorn-neo \
--datacard ${{dataset}}_{}_10000_float32 \
--l {} --r {} --k 10 {} {}'''
              inner = '\n'.join([
                inner_tmpl.format(
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

  da_s = [1, 2]
  spans = [30, 20]
  types = ["skewed", "correlated"]
  for m in ["CompassPostKTh", "ACORN"]:
    k_s = [10]
    idx = 0
    for group, datasets in M_GROUP_DATASET[m].items():
      parts = [p.lower() for p in re.findall(r'[A-Z][a-z]*', ("Acorn" if m == "ACORN" else m))]
      for d in datasets:
        for M in D_ARGS[d].get("M", M_ARGS[m].get("M", [1])):
          idx += 1
          filename = "revision-" + "-".join(parts) + f"-exp-{idx}.sh"
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

            workloads = zip(da_s, spans, types)
            inner_tmpl = \
'''/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-{}-revision \
--datacard ${{dataset}}_{}_{}_float32_{} --k ${{k}} {} {}'''
            inner = '\n'.join([
              inner_tmpl.format(
                "-".join(parts),
                *wl,
                build_string,
                search_string,
              ) for wl in workloads
            ])

            for bp in M_PARAM[m]["build"][::-1]:
              inner = __enclose_for(bp, f"{bp}_s", inner)
            # if m == "CompassPostKTh" and da == 1:
            #   k_s = [5, 10, 15, 20, 25]
            inner = __enclose_for("k", "k_s", inner)
            inner = (f'k_s=({" ".join(map(str, k_s))})\n') + inner
            f.write(inner)
            f.write('\n')

  for m in ["CompassPostKTh", "ACORN"]:
    k_s = [10]
    idx = 0
    for group, datasets in M_GROUP_DATASET[m].items():
      parts = [p.lower() for p in re.findall(r'[A-Z][a-z]*', ("Acorn" if m == "ACORN" else m))]
      for d in datasets:
        for M in D_ARGS[d].get("M", M_ARGS[m].get("M", [1])):
          idx += 1
          filename = "revision-filter-" + "-".join(parts) + f"-exp-{idx}.sh"
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

            workloads = ["onesided", "point", "negation"]
            inner_tmpl = \
'''/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-{}-revision-filter \
--datacard ${{dataset}}_{}_{}_float32_{} --k ${{k}} {} {}'''
            inner = '\n'.join([
              inner_tmpl.format(
                "-".join(parts),
                1,
                30,
                wl,
                build_string,
                search_string,
              ) for wl in workloads
            ])

            for bp in M_PARAM[m]["build"][::-1]:
              inner = __enclose_for(bp, f"{bp}_s", inner)
            # if m == "CompassPostKTh" and da == 1:
            #   k_s = [5, 10, 15, 20, 25]
            inner = __enclose_for("k", "k_s", inner)
            inner = (f'k_s=({" ".join(map(str, k_s))})\n') + inner
            f.write(inner)
            f.write('\n')


if __name__ == "__main__":
  # compose()
  compose_revision()
