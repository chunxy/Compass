#include "utils/reader.h"
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

FVecItrReader::FVecItrReader(std::string dataset_path) {
  ifs_.open(dataset_path.c_str(), std::ios::binary & std::ios::in);
  assert(ifs_.is_open());
  eof_flag_ = false;
  Next();
}

std::vector<float> FVecItrReader::Next() {
  auto ret = curr_;
  uint32_t d;
  if (!eof_flag_ && ifs_.read((char *)&d, sizeof(int))) {
    curr_.resize(d);
    ifs_.read((char *)curr_.data(), sizeof(float) * d);
  } else {
    curr_.resize(0);
    eof_flag_ = true;
  }
  return ret;
}

bool FVecItrReader::HasEnded() { return eof_flag_; }

IVecItrReader::IVecItrReader(std::string dataset_path) {
  ifs_.open(dataset_path.c_str(), std::ios::binary & std::ios::in);
  assert(ifs_.is_open());
  eof_flag_ = false;
  Next();
}

std::vector<uint32_t> IVecItrReader::Next() {
  auto ret = curr_;
  uint32_t d;
  if (!eof_flag_ && ifs_.read((char *)&d, sizeof(d))) {
    curr_.resize(d);
    ifs_.read((char *)curr_.data(), sizeof(uint32_t) * d);
  } else {
    curr_.resize(0);
    eof_flag_ = true;
  }
  return ret;
}

bool IVecItrReader::HasEnded() { return eof_flag_; }
