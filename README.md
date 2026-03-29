# Compass

Research-oriented C++ codebase for **hybrid** approximate nearest-neighbor experiments: vector similarity combined with attribute constraints. This README is meant to orient newcomers; algorithmic details stay in the code and papers.

---

## Repository layout

```text
Compass/
├── CMakeLists.txt                 # Top-level build: C++20, OpenMP, Boost, Faiss subtree, targets under src/
├── include/                       # Public headers: methods, helpers, and path templates
│   ├── config.h                   # Workspace-relative dirs (see below)
│   ├── methods/                   # Index / search method interfaces and implementations
│   ├── hnswlib/                   # Graph ANN primitives used by several benchmarks
│   └── utils/                     # Shared utilities (cards, I/O helpers, etc.)
├── src/
│   ├── benchmarks/                # Standalone timing / evaluation executables (CLI)
│   ├── apps/                      # One-off tools (e.g., ground-truth prep)
│   ├── tests/                     # Unit / regression tests
│   ├── utils/                     # Sources for shared utilities (e.g., workload registry)
│   └── acorn/                     # Baseline / comparison code paths
├── scripts/                       # Python helpers for data prep, clustering, plotting (optional)
├── thirdparty/                    # Vendored deps (Faiss, btree, …); built as part of CMake
├── .vscode/                       # Editor launch configs for common debug sessions
├── checkpoints/                   # Runtime index snapshots (created by benchmarks; layout from config)
├── data/                          # Expected layout for vectors, attributes, ground truth (see config.h)
├── logs_*/                        # Benchmark output logs (pattern from config.h)
└── stats/                         # Aggregated run statistics (if used by your workflow)
```

Paths like `checkpoints/`, `data/`, and `logs_*` are **not** exhaustive in git; they are populated when you run tools. Adjust `WORKSPACE` and directory constants in `include/config.h` for your machine.


## Main experiment driver: `bench-compass-post-k-th`

`src/benchmarks/bench-compass-post-k-th.cpp` builds the composite index (IVF, ranking, HNSW, cluster graph as applicable), runs **post-filtered** batched search with tunable search parameters, and writes structured logs (text + JSON) under the log root from `config.h`.

**Typical invocation**: run the binary from the repository root, passing a **datacard** name and index/search hyperparameters. Example:

```bash
./build/Debug/src/benchmarks/bench-compass-post-k-th \
  --datacard sift-dedup_1_30_float32_skewed \
  --k 10 \
  --M 16 \
  --efc 200 \
  --nlist 5000 \
  --efs 800 \
  --nrel 50 \
  --M_cg 4 \
  --batch_k 20 \
  --initial_efs 20 \
  --delta_efs 20
```

Replace `build/Debug/...` with your CMake output directory (`Release`, etc.). The `--datacard` must name a workload registered in `src/utils/cards.cpp` (see below). Other benchmarks in `src/benchmarks/` follow similar CLI patterns.


## Configuration: `include/config.h`

`include/config.h` centralizes **where** the tree expects data and artifacts: workspace root, log and stats roots, raw/attribute/ground-truth/range-query paths, and **filename templates** for IVF/graph/cluster checkpoints and related files. Edit this file so paths match your environment before large runs.


## Workloads: `src/utils/cards.cpp`

`src/utils/cards.cpp` defines the **datacards**: named bundles of vector paths, dimensions, counts, attribute metadata, and types. Benchmarks resolve `--datacard` through this registry; adding a new experiment usually means adding a card (and ensuring on-disk files match `config.h` layouts).


## Build (sketch)

Configure and build with CMake (dependencies: Boost, fmt, OpenMP; Faiss builds from `thirdparty/`). Point `include/config.h` at your workspace and data layout, then build your chosen target (e.g. `bench-compass-post-k-th`) using CMake.


## Note on sharing

This repository is shared to help others **navigate** the layout and reproduce high-level workflows. It is not a full product or paper supplement; treat internals as research code.
