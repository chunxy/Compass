link_libraries(Boost::filesystem)
link_libraries(roaring)

# Baseline
add_executable(bench-compass-1d-ivf bench-compass-1d-ivf.cpp)
add_executable(bench-compass-1d-ivf-imi bench-compass-1d-ivf-imi.cpp)
add_executable(bench-compass-1d-ivf-kmeanspp bench-compass-1d-ivf-kmeans++.cpp)
add_executable(bench-compass-1d-ivf-bikmeans bench-compass-1d-ivf-bikmeans.cpp)
add_executable(bench-compass-graph-1d bench-compass-graph-1d.cpp)
add_executable(bench-hop-hnsw-1d bench-hop-hnsw-1d.cpp)
add_executable(bench-postfiltering bench-postfiltering.cpp)
add_executable(bench-prefiltering-1d bench-prefiltering-1d.cpp)
add_executable(bench-compass-1d-post-k bench-compass-1d-post-k.cpp)
add_executable(bench-compass-1d-post-k-th bench-compass-1d-post-k-th.cpp)
add_executable(bench-compass-1d-post-k-navix bench-compass-1d-post-k-navix.cpp)
add_executable(bench-compass-post-k-th bench-compass-post-k-th.cpp)
add_executable(bench-compass-post-k-th-revision bench-compass-post-k-th-revision.cpp)
add_executable(bench-compass-post-k-th-revision-filter bench-compass-post-k-th-revision-filter.cpp)
add_executable(bench-compass-post-k-thb bench-compass-post-k-thb.cpp)
add_executable(bench-compass-post-k-th-bs bench-compass-post-k-th-bs.cpp)
add_executable(bench-compass-post-k-th-ch bench-compass-post-k-th-ch.cpp)
add_executable(bench-acorn bench-acorn.cpp)
add_executable(bench-acorn-neo bench-acorn-neo.cpp)
add_executable(bench-acorn-revision bench-acorn-revision.cpp)
add_executable(bench-acorn-revision-filter bench-acorn-revision-filter.cpp)
target_link_libraries(bench-acorn ACORN)
target_link_libraries(bench-acorn-neo ACORN)
target_link_libraries(bench-acorn-revision ACORN)
target_link_libraries(bench-acorn-revision-filter ACORN)
# add_executable(bench-compass-nsg bench-compass-nsg.cpp)

# Legacy
# add_executable(bench-compass-r-trick-1d bench-compass-r-trick-1d.cpp)
add_executable(bench-compass-1d-k-gr bench-compass-1d-k-gr.cpp)
add_executable(bench-compass-1d-k-or bench-compass-1d-k-or.cpp)
add_executable(bench-compass-1d-k-or-cg bench-compass-1d-k-or-cg.cpp)

# KMeans and variants
add_executable(bench-compass-1d-k bench-compass-1d-k.cpp)
add_executable(bench-compass-1d-k-cg bench-compass-1d-k-cg.cpp)
add_executable(bench-compass-1d-kmedoids bench-compass-1d-kmedoids.cpp)
add_executable(bench-compass-1d-kmedoids-cg bench-compass-1d-kmedoids-cg.cpp)
add_executable(bench-compass-1d-bikmeans bench-compass-1d-bikmeans.cpp)
add_executable(bench-compass-1d-bikmeans-cg bench-compass-1d-bikmeans-cg.cpp)
add_executable(bench-compass-1d-bikmeans-or bench-compass-1d-bikmeans-or.cpp)

# Decent clustering
add_executable(bench-compass-1d-pca bench-compass-1d-pca.cpp)
add_executable(bench-compass-1d-pca-cg bench-compass-1d-pca-cg.cpp)

# Novel clustering
add_executable(bench-compass-1d-llem bench-compass-1d-llem.cpp)
add_executable(bench-compass-1d-idec bench-compass-1d-idec.cpp)

# Iterative cluster graph
add_executable(bench-compass-1d-k-icg bench-compass-1d-k-icg.cpp)
add_executable(bench-compass-1d-k-qicg bench-compass-1d-k-qicg.cpp)
add_executable(bench-compass-1d-bikmeans-icg bench-compass-1d-bikmeans-icg.cpp)
add_executable(bench-compass-1d-bikmeans-qicg bench-compass-1d-bikmeans-qicg.cpp)
add_executable(bench-compass-1d-pca-icg bench-compass-1d-pca-icg.cpp)
add_executable(bench-compass-1d-pca-qicg bench-compass-1d-pca-qicg.cpp)

# Baseline
add_executable(bench-compass-ivf bench-compass-ivf.cpp)
add_executable(bench-compass-graph bench-compass-graph.cpp)
add_executable(bench-compass-relational bench-compass-relational.cpp)

# Legacy
add_executable(bench-compass-k-or bench-compass-k-or.cpp)
add_executable(bench-compass-k-or-cg bench-compass-k-or-cg.cpp)

# KMeans and variants
add_executable(bench-compass-k bench-compass-k.cpp)
add_executable(bench-compass-k-cg bench-compass-k-cg.cpp)
add_executable(bench-compass-bikmeans bench-compass-bikmeans.cpp)
add_executable(bench-compass-bikmeans-cg bench-compass-bikmeans-cg.cpp)

# Decent clustering
add_executable(bench-compass-pca bench-compass-pca.cpp)
add_executable(bench-compass-pca-cg bench-compass-pca-cg.cpp)

# Iterative cluster graph
add_executable(bench-compass-k-icg bench-compass-k-icg.cpp)
add_executable(bench-compass-k-qicg bench-compass-k-qicg.cpp)
add_executable(bench-compass-bikmeans-icg bench-compass-bikmeans-icg.cpp)
add_executable(bench-compass-bikmeans-qicg bench-compass-bikmeans-qicg.cpp)
add_executable(bench-compass-pca-icg bench-compass-pca-icg.cpp)
add_executable(bench-compass-pca-qicg bench-compass-pca-qicg.cpp)
