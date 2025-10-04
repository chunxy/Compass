#include <faiss/IndexBinaryFromFloat.h>
#include <faiss/index_io.h>

namespace acorn {

using namespace faiss;

void write_acorn_index(const Index *idx, const char *fname);
Index *read_acorn_index(const char *fname);
}  // namespace acorn