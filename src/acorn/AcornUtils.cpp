/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/index_io.h>

#include <cstdio>
#include <cstdlib>
#include "acorn/ACORN.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <faiss/invlists/InvertedListsIOHook.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>
#include <faiss/utils/hamming.h>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryFromFloat.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexBinaryHash.h>
#include <faiss/IndexBinaryIVF.h>

// hybrid indices
#include "acorn/AcornUtils.h"
#include "acorn/IndexACORN.h"

/*************************************************************
 * The I/O format is the content of the class. For objects that are
 * inherited, like Index, a 4-character-code (fourcc) indicates which
 * child class this is an instance of.
 *
 * In this case, the fields of the parent class are written first,
 * then the ones for the child classes. Note that this requires
 * classes to be serialized to have a constructor without parameters,
 * so that the fields can be filled in later. The default constructor
 * should set reasonable defaults for all fields.
 *
 * The fourccs are assigned arbitrarily. When the class changed (added
 * or deprecated fields), the fourcc can be replaced. New code should
 * be able to read the old fourcc and fill in new classes.
 *
 * TODO: in this file, the read functions that encouter errors may
 * leak memory.
 **************************************************************/

namespace acorn {

using namespace faiss;

static void write_index_header(const Index *idx, IOWriter *f) {
  WRITE1(idx->d);
  WRITE1(idx->ntotal);
  idx_t dummy = 1 << 20;
  WRITE1(dummy);
  WRITE1(dummy);
  WRITE1(idx->is_trained);
  WRITE1(idx->metric_type);
  if (idx->metric_type > 1) {
    WRITE1(idx->metric_arg);
  }
}

static void read_index_header(Index *idx, IOReader *f) {
  READ1(idx->d);
  READ1(idx->ntotal);
  idx_t dummy;
  READ1(dummy);
  READ1(dummy);
  READ1(idx->is_trained);
  READ1(idx->metric_type);
  if (idx->metric_type > 1) {
    READ1(idx->metric_arg);
  }
  idx->verbose = false;
}

static void write_ACORN(const ACORN *hnsw, IOWriter *f) {
  WRITEVECTOR(hnsw->assign_probas);
  WRITEVECTOR(hnsw->cum_nneighbor_per_level);
  WRITEVECTOR(hnsw->levels);
  WRITEVECTOR(hnsw->offsets);
  WRITEVECTOR(hnsw->neighbors);

  // added for hybrid version
  WRITEVECTOR(hnsw->nb_per_level)
  // WRITEVECTOR(hnsw->metadata)

  WRITE1(hnsw->entry_point);
  WRITE1(hnsw->max_level);
  WRITE1(hnsw->efConstruction);
  WRITE1(hnsw->efSearch);
  WRITE1(hnsw->upper_beam);

  // added for hybrid version
  WRITE1(hnsw->gamma);
  WRITE1(hnsw->M);
  WRITE1(hnsw->M_beta);
}

static void read_ACORN(ACORN *acorn, IOReader *f) {
  READVECTOR(acorn->assign_probas);
  READVECTOR(acorn->cum_nneighbor_per_level);
  READVECTOR(acorn->levels);
  READVECTOR(acorn->offsets);
  READVECTOR(acorn->neighbors);

  // added for hybrid version
  READVECTOR(acorn->nb_per_level);
  // READVECTOR(acorn->metadata);

  READ1(acorn->entry_point);
  READ1(acorn->max_level);
  READ1(acorn->efConstruction);
  READ1(acorn->efSearch);
  READ1(acorn->upper_beam);
  READ1(acorn->gamma);
  READ1(acorn->M);
  READ1(acorn->M_beta);
}

void write_acorn_index(const Index *idx, const char *fname) {
  FileIOWriter writer(fname);
  FileIOWriter *f = &writer;
  if (const IndexACORN *indxacorn = dynamic_cast<const IndexACORN *>(idx)) {
    uint32_t h = fourcc("IHNH");  // this needs to be a 4 letter header
    FAISS_THROW_IF_NOT(h != 0);
    WRITE1(h);
    write_index_header(indxacorn, f);
    write_ACORN(&indxacorn->acorn, f);
    write_index(indxacorn->storage, f);
  }
}

Index *read_acorn_index(const char *fname) {
  FileIOReader reader(fname);
  FileIOReader *f = &reader;
  Index *idx = nullptr;
  uint32_t h;
  READ1(h);
  if (h == fourcc("IHNH")) {
    // IndexHNSWFlat* idxhnswhybrid = new IndexHNSWFlat();
    IndexACORN *idxacorn = nullptr;
    std::vector<int> metadata = {};
    idxacorn = new IndexACORNFlat(0, 0, 0, metadata, 0);
    // IndexACORNFlat* idxhnsw = new IndexACORNFlat();
    // IndexACORN* idxhnsw = new IndexACORNFlat();
    read_index_header(idxacorn, f);
    read_ACORN(&idxacorn->acorn, f);
    idxacorn->storage = read_index(f);
    idxacorn->own_fields = true;
    idx = idxacorn;
  }
  return idx;
}

}  // namespace acorn
