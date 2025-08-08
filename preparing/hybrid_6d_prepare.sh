datasets=(sift gist crawl glove100 video audio)

# Generate hybrid attributes
for dataset in ${datasets[@]}; do
  /home/chunxy/repos/Compass/build/Release/src/apps/generate-attributes --datacard ${dataset}_6_10000_float32
done

