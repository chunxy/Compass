datasets=(sift glove100 audio crawl gist video)
M_s=(16 32)
nlist_s=(1000 5000 10000)
r1_s=(200 600 1100 3100 5100 8100 9100)
r2_s=(300 700 1200 3200 5200 8200 9200)
efs_s=(10 20 60 100 200 300 400 500)
nrel_s=(100 200)

for d in ${datasets[@]}; do
  for M in ${M_s[@]}; do
    for nlist in ${nlist_s[@]}; do
      for r1 in ${r1_s[@]}; do
        for r2 in ${r2_s[@]}; do
          /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_r --datacard ${d}_2_10000_float32 \
          --l 100 200 --r ${r1} ${r2} --k 10 --M ${M} --efc 200 --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
        done
      done
    done
  done
done
