datasets=(video glove100)
M_s=(16 32)
nlist_s=(10000 20000)
r1_s=(200 600 1100 3100 5100 8100 9100)
r2_s=(300 700 1200 3200 5200 8200 9200)
efs_s=(10 20 60 100 200)
nrel_s=(100 200)

for dataset in ${datasets[@]}; do
  for M in ${M_s[@]}; do
    for nlist in ${nlist_s[@]}; do
      i=0
      while [ $i -lt ${#r1_s[@]} ]; do
        r1=${r1_s[$i]}
        r2=${r2_s[$i]}
        /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-bikmeans --datacard ${dataset}_2_10000_float32 \
          --l 100 200 --r ${r1} ${r2} --k 10 --M ${M} --efc 200 --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
        i=$((i + 1))
      done
    done
  done
done
