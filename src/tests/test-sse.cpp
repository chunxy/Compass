#include <iostream>
#include "hnswlib/hnswlib.h"

template <typename T>
class Container {
 public:
  template <bool isList = true>
  auto CreateContainer() {
    if constexpr (isList) {
      return std::list<T>{};
    } else {
      return std::vector<T>{};
    }
  }
};

int main() {
#ifdef __AVX512F__
  std::cout << "__AVX512F__ enabled" << std::endl;
#endif

#ifdef __AVX__
  std::cout << "__AVX__ enabled" << std::endl;
#endif

#ifdef __SSE__
  std::cout << "__SSE__ enabled" << std::endl;
#endif

#ifdef USE_AVX
  std::cout << "USE_AVX enabled" << std::endl;
#endif

#ifdef USE_SSE
  std::cout << "USE_SSE enabled" << std::endl;
#endif
  auto listContainer = Container<int>().CreateContainer();
  auto vectorContainer = Container<int>().CreateContainer<false>();

  std::cout << "List container created with size: " << listContainer.size() << std::endl;
  std::cout << "Vector container created with size: " << vectorContainer.size() << std::endl;
}