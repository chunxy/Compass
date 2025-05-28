datasets=(sift gist crawl glove100 video audio)
M_s=(32)
nlist_s=(1000 2000 5000 10000)
r_s=(200 300 600 1100 2100 3100 4100 5100 6100 7100 8100 9100)
efs_s=(10 20 60 100 200)
nrel_s=(100 200 500)

for dataset in ${datasets[@]}; do
  for M in ${M_s[@]}; do
    for nlist in ${nlist_s[@]}; do
      for r in ${r_s[@]}; do
        /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_r_1d --datacard ${dataset}_1_10000_float32 --l 100 --r ${r} --k 10 --M ${M} --efc 200 --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
      done
      /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_r_1d --datacard ${dataset}_1_10000_float32 --l 0 --r 10000 --k 10 --M ${M} --efc 200 --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
    done
  done
done

M_s=(16)
nlist_s=(1000)
for dataset in ${datasets[@]}; do
  for M in ${M_s[@]}; do
    for nlist in ${nlist_s[@]}; do
      for r in ${r_s[@]}; do
        /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_r_1d --datacard ${dataset}_1_10000_float32 --l 100 --r ${r} --k 10 --M ${M} --efc 200 --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
      done
      /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_r_1d --datacard ${dataset}_1_10000_float32 --l 0 --r 10000 --k 10 --M ${M} --efc 200 --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
    done
  done
done

# /home/chunxy/repos/Compass/build/Release/src/benchmarks/bench_compass_r_old_1d --datacard video_1_10000_float32 --l 100 --r 200 --k 10 --M 16 --efc 200 --nlist 1000 --efs 100 --nrel 100
