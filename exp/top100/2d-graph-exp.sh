datasets=(sift gist crawl glove100 audio video)
r1_s=(1100 3100 5100 8100 9100)
r2_s=(1200 3200 5200 8200 9200)
M_s=(32)
efs_s=(100 200)
nrel_s=(100 200 500)

for dataset in ${datasets[@]}; do
  for M in ${M_s[@]}; do
    for nlist in ${nlist_s[@]}; do
      i=0
      while [ $i -lt ${#r1_s[@]} ]; do
        r1=${r1_s[$i]}
        r2=${r2_s[$i]}
        /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_graph \
          --datacard ${dataset}_2_10000_float32 --l 100 200 --r ${r1} ${r2} --k 100 \
          --M ${M} --efc 200 --efs ${efs_s[@]} --nrel ${nrel_s[@]}
        i=$((i + 1))
      done
    done
  done
done