datasets=(sift gist audio)
M_s=(16 32)
efc_s=(200)
nlist_s=(10000 20000)
dx_s=(64)
efs_s=(10 20 60 100 200 300)
nrel_s=(100 200)
for dataset in ${datasets[@]}; do
for M in ${M_s[@]}; do
for efc in ${efc_s[@]}; do
for nlist in ${nlist_s[@]}; do
for dx in ${dx_s[@]}; do
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-pca --datacard ${dataset}_2_10000_float32 --l 100 200 --r 1100 1200 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --dx ${dx} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-pca --datacard ${dataset}_2_10000_float32 --l 100 200 --r 3100 3200 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --dx ${dx} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-pca --datacard ${dataset}_2_10000_float32 --l 100 200 --r 5100 5200 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --dx ${dx} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-pca --datacard ${dataset}_2_10000_float32 --l 100 200 --r 8100 8200 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --dx ${dx} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
/home/chunxy/repos/Compass/build/Release/src/benchmarks/bench-compass-pca --datacard ${dataset}_2_10000_float32 --l 100 200 --r 9100 9200 --k 10 --M ${M} --efc ${efc} --nlist ${nlist} --dx ${dx} --efs ${efs_s[@]} --nrel ${nrel_s[@]}
done
done
done
done
done