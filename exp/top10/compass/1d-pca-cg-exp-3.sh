datasets=(sift audio gist)
dpca_s=(64 64)
M_s=(16 32)
nlist_s=(10000 20000)
r_s=(200 300 600 1100 2100 3100 4100 5100 6100 7100 8100 9100)
efs_s=(10 20 60 100 200)
nrel_s=(100 200)

i=0
while [ $i -lt ${#datasets[@]} ]; do
  dataset=${datasets[$i]}
  dpca=${dpca_s[$i]}
  i=$((i+1))
  for M in ${M_s[@]}; do
    for nlist in ${nlist_s[@]}; do
      for r in ${r_s[@]}; do
        /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_1d_pca_cg --datacard ${dataset}_1_10000_float32 --l 100 --r ${r} --k 10 --M ${M} --efc 200 --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]} --dx ${dpca}
      done
      /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_1d_pca_cg --datacard ${dataset}_1_10000_float32 --l 0 --r 10000 --k 10 --M ${M} --efc 200 --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]} --dx ${dpca}
    done
  done
done

datasets=(gist)
dpca_s=(512)
nlist_s=(10000 20000)

i=0
while [ $i -lt ${#datasets[@]} ]; do
  dataset=${datasets[$i]}
  dpca=${dpca_s[$i]}
  i=$((i+1))
  for M in ${M_s[@]}; do
    for nlist in ${nlist_s[@]}; do
      for r in ${r_s[@]}; do
        /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-1d-pca-cg --datacard ${dataset}_1_10000_float32 --l 100 --r ${r} --k 10 --M ${M} --efc 200 --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]} --dx ${dpca}
      done
      /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-1d-pca-cg --datacard ${dataset}_1_10000_float32 --l 0 --r 10000 --k 10 --M ${M} --efc 200 --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]} --dx ${dpca}
    done
  done
done

# /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_rr_cg_1d --datacard sift_1_10000_float32 --l 100 --r 200 --k 10 --M 16 --efc 200 --nlist 10000 --efs 100 --nrel 100

