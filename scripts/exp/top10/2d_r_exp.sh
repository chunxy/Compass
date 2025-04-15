datasets=(sift gist crawl glove100 video audio)
M_s=(16 32)
nlist_s=(1000 5000 10000)
r_s=("1100 1200" "3100 3200" "5100 5200" "8100 8200" "9100 9200")
efs_s=(10 20 60 100 120 140 160 180 200 220 240 260 280 300)
nrel_s=(100 200)

for dataset in ${datasets[@]}; do
  for M in ${M_s[@]}; do
    for nlist in ${nlist_s[@]}; do
      for r in "${r_s[@]}"; do
        /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_r --datacard ${dataset}_2_10000_float32 \
        --l 100 200 --r ${r} --k 10 --M ${M} --efc 200 --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
      done
    done
  done
done