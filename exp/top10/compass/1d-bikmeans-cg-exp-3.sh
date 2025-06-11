datasets=(sift gist audio)
M_s=(16 32)
efc_s=(200)
nlist_s=(10000 20000)
efs_s=(10 20 60 100 200 300)
nrel_s=(100 200)
for dataset in ${datasets[@]}; do
for M in ${M_s[@]}; do
for efc in ${efc_s[@]}; do
for nlist in ${nlist_s[@]}; do
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-1d-bikmeans-cg --datacard ${dataset}_1_10000_float32 --l 100 --r 200 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-1d-bikmeans-cg --datacard ${dataset}_1_10000_float32 --l 100 --r 300 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-1d-bikmeans-cg --datacard ${dataset}_1_10000_float32 --l 100 --r 600 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-1d-bikmeans-cg --datacard ${dataset}_1_10000_float32 --l 100 --r 1100 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-1d-bikmeans-cg --datacard ${dataset}_1_10000_float32 --l 100 --r 2100 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-1d-bikmeans-cg --datacard ${dataset}_1_10000_float32 --l 100 --r 3100 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-1d-bikmeans-cg --datacard ${dataset}_1_10000_float32 --l 100 --r 4100 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-1d-bikmeans-cg --datacard ${dataset}_1_10000_float32 --l 100 --r 5100 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-1d-bikmeans-cg --datacard ${dataset}_1_10000_float32 --l 100 --r 6100 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-1d-bikmeans-cg --datacard ${dataset}_1_10000_float32 --l 100 --r 7100 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-1d-bikmeans-cg --datacard ${dataset}_1_10000_float32 --l 100 --r 8100 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-1d-bikmeans-cg --datacard ${dataset}_1_10000_float32 --l 100 --r 9100 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-1d-bikmeans-cg --datacard ${dataset}_1_10000_float32 --l 0 --r 10000 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
done
done
done
done