#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#define CHECK_REGEX(string, regex) (!string.empty() && std::isdigit(string[0]))

enum Operation {
  EQUAL = 0,
  OR = 1,
  REGEX = 2,
};

struct VisitedTable {
  std::vector<uint8_t> visited;
  int visno;

  explicit VisitedTable(int size) : visited(size), visno(1) {}

  /// set flag #no to true
  void set(int no) { visited[no] = visno; }

  /// get flag #no
  bool get(int no) const { return visited[no] == visno; }

  /// reset all flags to false
  void advance() {
    visno++;
    if (visno == 250) {
      // 250 rather than 255 because sometimes we use visno and visno+1
      memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
      visno = 1;
    }
  }

  // added for hybrid search
  int num_visited() {
    int num = 0;
    for (int i = 0; i < visited.size(); i++) {
      if (visited[i] == visno) {
        num = num + 1;
      }
    }
    return num;
  }
};

namespace faiss {

/** bare-bones unique_ptr
 * this one deletes with delete [] */
template <class T>
struct ScopeDeleter {
    const T* ptr;
    explicit ScopeDeleter(const T* ptr = nullptr) : ptr(ptr) {}
    void release() {
        ptr = nullptr;
    }
    void set(const T* ptr_in) {
        ptr = ptr_in;
    }
    void swap(ScopeDeleter<T>& other) {
        std::swap(ptr, other.ptr);
    }
    ~ScopeDeleter() {
        delete[] ptr;
    }
};

template <class T>
struct ScopeDeleter1 {
  const T *ptr;
  explicit ScopeDeleter1(const T *ptr = nullptr) : ptr(ptr) {}
  void release() { ptr = nullptr; }
  void set(const T *ptr_in) { ptr = ptr_in; }
  void swap(ScopeDeleter1<T> &other) { std::swap(ptr, other.ptr); }
  ~ScopeDeleter1() { delete ptr; }
};
}  // namespace faiss