datasets=(sift audio glove100)
dpca_s=(64 64 64)
M_s=(16 32)
nlist_s=(10000 20000)
r1_s=(200 600 1100 3100 5100 8100 9100)
r2_s=(300 700 1200 3200 5200 8200 9200)
efs_s=(10 20 60 100 200)
nrel_s=(100 200)

i=0
while [ $i -lt ${#datasets[@]} ]; do
  dataset=${datasets[$i]}
  dpca=${dpca_s[$i]}
  i=$((i+1))
  for M in ${M_s[@]}; do
    for nlist in ${nlist_s[@]}; do
      j=0
      while [ $j -lt ${#r1_s[@]} ]; do
        r1=${r1_s[$j]}
        r2=${r2_s[$j]}
        /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-pca --datacard ${dataset}_2_10000_float32 \
          --l 100 200 --r ${r1} ${r2} --k 10 --M ${M} --efc 200 --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]} --dx ${dpca}
        j=$((j + 1))
      done
    done
  done
done

# /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_rr_cg_1d --datacard sift_1_10000_float32 --l 100 --r 200 --k 10 --M 16 --efc 200 --nlist 10000 --efs 100 --nrel 100

