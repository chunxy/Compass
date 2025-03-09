#pragma once

#include "hnswlib.h"
#include "visited_list_pool.h"
#include <assert.h>
#include <atomic>
#include <memory>
#include <stdlib.h>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template <typename dist_t> class Rng : public AlgorithmInterface<dist_t> {
public:
  static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
  static const unsigned char DELETE_MARK = 0x01;

  size_t max_elements_{0};
  mutable std::atomic<size_t> cur_element_count{
      0}; // current number of elements
  size_t size_data_per_element_{0};
  size_t size_links_per_element_{0};
  size_t M_{0};
  size_t maxM_{0};
  size_t maxM0_{0};
  size_t ef_construction_{0};
  size_t ef_{0};

  std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

  tableint enterpoint_node_{0};

  size_t size_links_level0_{0};
  size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

  char *data_level0_memory_{nullptr};

  size_t data_size_{0};

  DISTFUNC<dist_t> fstdistfunc_;
  void *dist_func_param_{nullptr};

  mutable std::atomic<long> metric_distance_computations{0};
  mutable std::atomic<long> metric_hops{0};

  Rng(SpaceInterface<dist_t> *s) {}

  Rng(SpaceInterface<dist_t> *s, const std::string &location,
      bool nmslib = false, size_t max_elements = 0) {
    loadIndex(location, s, max_elements);
  }

  Rng(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16,
      size_t ef_construction = 200, size_t random_seed = 100) {
    max_elements_ = max_elements;
    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    if (M <= 10000) {
      M_ = M;
    } else {
      HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse "
                 "effects."
              << std::endl;
      HNSWERR << "         Cap to 10000 will be applied for the rest of the "
                 "processing."
              << std::endl;
      M_ = 10000;
    }
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    size_data_per_element_ =
        size_links_level0_ + data_size_ + sizeof(labeltype);
    offsetData_ = size_links_level0_;
    label_offset_ = size_links_level0_ + data_size_;
    offsetLevel0_ = 0;

    data_level0_memory_ =
        (char *)malloc(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
      throw std::runtime_error("Not enough memory");

    cur_element_count = 0;

    visited_list_pool_ =
        std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;

    size_links_per_element_ =
        maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
  }

  ~Rng() { clear(); }

  void clear() {
    free(data_level0_memory_);
    data_level0_memory_ = nullptr;
    cur_element_count = 0;
    visited_list_pool_.reset(nullptr);
  }

  struct CompareByFirst {
    constexpr bool
    operator()(std::pair<dist_t, tableint> const &a,
               std::pair<dist_t, tableint> const &b) const noexcept {
      return a.first < b.first;
    }
  };

  void setEf(size_t ef) { ef_ = ef; }

  inline labeltype *getExternalLabeLp(tableint internal_id) const {
    return (labeltype *)(data_level0_memory_ +
                         internal_id * size_data_per_element_ + label_offset_);
  }

  inline char *getDataByInternalId(tableint internal_id) const {
    return (data_level0_memory_ + internal_id * size_data_per_element_ +
            offsetData_);
  }

  size_t getMaxElements() { return max_elements_; }

  size_t getCurrentElementCount() { return cur_element_count; }

  std::priority_queue<std::pair<dist_t, tableint>,
                      std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
  searchBaseLayer(tableint ep_id, const void *data_point) {
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        candidateSet;

    dist_t lowerBound;
    dist_t dist =
        fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
    top_candidates.emplace(dist, ep_id);
    lowerBound = dist;
    candidateSet.emplace(-dist, ep_id);

    visited_array[ep_id] = visited_array_tag;

    while (!candidateSet.empty()) {
      std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
      if ((-curr_el_pair.first) > lowerBound &&
          top_candidates.size() == ef_construction_) {
        break;
      }
      candidateSet.pop();

      tableint curNodeNum = curr_el_pair.second;

      linklistsizeint *data = get_linklist0(curNodeNum);
      size_t size = getListCount((linklistsizeint *)data);
      tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
      _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
      _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
      _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

      for (size_t j = 0; j < size; j++) {
        tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
        _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
        if (visited_array[candidate_id] == visited_array_tag)
          continue;
        visited_array[candidate_id] = visited_array_tag;
        char *currObj1 = (getDataByInternalId(candidate_id));

        dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
        if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
          candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
          _mm_prefetch(getDataByInternalId(candidateSet.top().second),
                       _MM_HINT_T0);
#endif

          top_candidates.emplace(dist1, candidate_id);

          if (top_candidates.size() > ef_construction_)
            top_candidates.pop();

          if (!top_candidates.empty())
            lowerBound = top_candidates.top().first;
        }
      }
    }
    visited_list_pool_->releaseVisitedList(vl);

    return top_candidates;
  }

  // bare_bone_search means there is no check for deletions and stop condition
  // is ignored in return of extra performance
  template <bool collect_metrics = false>
  std::priority_queue<std::pair<dist_t, tableint>,
                      std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
  searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef,
                    BaseFilterFunctor *isIdAllowed = nullptr) const {
    VisitedList *vl = visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        candidate_set;

    dist_t lowerBound;
    if ((!isIdAllowed) || (*isIdAllowed)(ep_id)) {
      char *ep_data = getDataByInternalId(ep_id);
      dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
      lowerBound = dist;
      top_candidates.emplace(dist, ep_id);
      candidate_set.emplace(-dist, ep_id);
    } else {
      lowerBound = std::numeric_limits<dist_t>::max();
      candidate_set.emplace(-lowerBound, ep_id);
    }

    visited_array[ep_id] = visited_array_tag;

    while (!candidate_set.empty()) {
      std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
      dist_t candidate_dist = -current_node_pair.first;

      bool flag_stop_search =
          candidate_dist > lowerBound && top_candidates.size() == ef;
      if (flag_stop_search) {
        break;
      }
      candidate_set.pop();

      tableint current_node_id = current_node_pair.second;
      int *data = (int *)get_linklist0(current_node_id);
      size_t size = getListCount((linklistsizeint *)data);

      if (collect_metrics) {
        metric_hops++;
        metric_distance_computations += size;
      }

#ifdef USE_SSE
      _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
      _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
      _mm_prefetch(data_level0_memory_ +
                       (*(data + 1)) * size_data_per_element_ + offsetData_,
                   _MM_HINT_T0);
      _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

      for (size_t j = 1; j <= size; j++) {
        int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
        _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
        _mm_prefetch(data_level0_memory_ +
                         (*(data + j + 1)) * size_data_per_element_ +
                         offsetData_,
                     _MM_HINT_T0); ////////////
#endif
        if (!(visited_array[candidate_id] == visited_array_tag)) {
          visited_array[candidate_id] = visited_array_tag;

          char *currObj1 = (getDataByInternalId(candidate_id));
          dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

          bool flag_consider_candidate =
              top_candidates.size() < ef || lowerBound > dist;
          if (flag_consider_candidate) {
            candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
            _mm_prefetch(data_level0_memory_ +
                             candidate_set.top().second *
                                 size_data_per_element_ +
                             offsetLevel0_, ///////////
                         _MM_HINT_T0);      ////////////////////////
#endif

            if ((!isIdAllowed) || (*isIdAllowed)(candidate_id)) {
              top_candidates.emplace(dist, candidate_id);
            }

            while (top_candidates.size() > ef) {
              top_candidates.pop();
            }

            if (!top_candidates.empty())
              lowerBound = top_candidates.top().first;
          }
        }
      }
    }

    visited_list_pool_->releaseVisitedList(vl);
    return top_candidates;
  }

  void getNeighborsByHeuristic2(
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst> &top_candidates,
      const size_t M) {
    if (top_candidates.size() < M) {
      return;
    }

    std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
    std::vector<std::pair<dist_t, tableint>> return_list;
    while (top_candidates.size() > 0) {
      queue_closest.emplace(-top_candidates.top().first,
                            top_candidates.top().second);
      top_candidates.pop();
    }

    while (queue_closest.size()) {
      if (return_list.size() >= M)
        break;
      std::pair<dist_t, tableint> curent_pair = queue_closest.top();
      dist_t dist_to_query = -curent_pair.first;
      queue_closest.pop();
      bool good = true;

      for (std::pair<dist_t, tableint> second_pair : return_list) {
        dist_t curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
                                      getDataByInternalId(curent_pair.second),
                                      dist_func_param_);
        if (curdist < dist_to_query) {
          good = false;
          break;
        }
      }
      if (good) {
        return_list.push_back(curent_pair);
      }
    }

    for (std::pair<dist_t, tableint> curent_pair : return_list) {
      top_candidates.emplace(-curent_pair.first, curent_pair.second);
    }
  }

  linklistsizeint *get_linklist0(tableint internal_id) const {
    return (linklistsizeint *)(data_level0_memory_ +
                               internal_id * size_data_per_element_ +
                               offsetLevel0_);
  }

  linklistsizeint *get_linklist0(tableint internal_id,
                                 char *level0_memory_) const {
    return (linklistsizeint *)(level0_memory_ +
                               internal_id * size_data_per_element_ +
                               offsetLevel0_);
  }

  tableint mutuallyConnectNewElement(
      const void *data_point, tableint cur_c,
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst> &top_candidates) {
    size_t Mcurmax = maxM0_;
    getNeighborsByHeuristic2(top_candidates, M_);
    if (top_candidates.size() > M_)
      throw std::runtime_error(
          "Should be not be more than M_ candidates returned by the heuristic");

    std::vector<tableint> selectedNeighbors;
    selectedNeighbors.reserve(M_);
    while (top_candidates.size() > 0) {
      selectedNeighbors.push_back(top_candidates.top().second);
      top_candidates.pop();
    }

    tableint next_closest_entry_point = selectedNeighbors.back();

    {
      linklistsizeint *ll_cur = get_linklist0(cur_c);

      if (*ll_cur) {
        throw std::runtime_error(
            "The newly inserted element should have blank link list");
      }
      setListCount(ll_cur, selectedNeighbors.size());
      tableint *data = (tableint *)(ll_cur + 1);
      for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
        if (data[idx])
          throw std::runtime_error("Possible memory corruption");
        data[idx] = selectedNeighbors[idx];
      }
    }

    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

      linklistsizeint *ll_other = get_linklist0(selectedNeighbors[idx]);

      size_t sz_link_list_other = getListCount(ll_other);

      if (sz_link_list_other > Mcurmax)
        throw std::runtime_error("Bad value of sz_link_list_other");
      if (selectedNeighbors[idx] == cur_c)
        throw std::runtime_error("Trying to connect an element to itself");

      tableint *data = (tableint *)(ll_other + 1);

      bool is_cur_c_present = false;
      // If cur_c is already present in the neighboring connections of
      // `selectedNeighbors[idx]` then no need to modify any connections or run
      // the heuristics.
      if (!is_cur_c_present) {
        if (sz_link_list_other < Mcurmax) {
          data[sz_link_list_other] = cur_c;
          setListCount(ll_other, sz_link_list_other + 1);
        } else {
          // finding the "weakest" element to replace it with the new one
          dist_t d_max = fstdistfunc_(
              getDataByInternalId(cur_c),
              getDataByInternalId(selectedNeighbors[idx]), dist_func_param_);
          // Heuristic:
          std::priority_queue<std::pair<dist_t, tableint>,
                              std::vector<std::pair<dist_t, tableint>>,
                              CompareByFirst>
              candidates;
          candidates.emplace(d_max, cur_c);

          for (size_t j = 0; j < sz_link_list_other; j++) {
            candidates.emplace(
                fstdistfunc_(getDataByInternalId(data[j]),
                             getDataByInternalId(selectedNeighbors[idx]),
                             dist_func_param_),
                data[j]);
          }

          getNeighborsByHeuristic2(candidates, Mcurmax);

          int indx = 0;
          while (candidates.size() > 0) {
            data[indx] = candidates.top().second;
            candidates.pop();
            indx++;
          }

          setListCount(ll_other, indx);
          // Nearest K:
          /*int indx = -1;
          for (int j = 0; j < sz_link_list_other; j++) {
              dist_t d = fstdistfunc_(getDataByInternalId(data[j]),
          getDataByInternalId(rez[idx]), dist_func_param_); if (d > d_max) {
                  indx = j;
                  d_max = d;
              }
          }
          if (indx >= 0) {
              data[indx] = cur_c;
          } */
        }
      }
    }

    return next_closest_entry_point;
  }

  size_t indexFileSize() const {
    size_t size = 0;
    size += sizeof(offsetLevel0_);
    size += sizeof(max_elements_);
    size += sizeof(cur_element_count);
    size += sizeof(size_data_per_element_);
    size += sizeof(label_offset_);
    size += sizeof(offsetData_);
    size += sizeof(enterpoint_node_);
    size += sizeof(maxM_);

    size += sizeof(maxM0_);
    size += sizeof(M_);
    size += sizeof(ef_construction_);

    size += cur_element_count * size_data_per_element_;

    return size;
  }

  void saveIndex(const std::string &location) {
    std::ofstream output(location, std::ios::binary);
    std::streampos position;

    writeBinaryPOD(output, offsetLevel0_);
    writeBinaryPOD(output, max_elements_);
    writeBinaryPOD(output, cur_element_count);
    writeBinaryPOD(output, size_data_per_element_);
    writeBinaryPOD(output, label_offset_);
    writeBinaryPOD(output, offsetData_);
    writeBinaryPOD(output, enterpoint_node_);
    writeBinaryPOD(output, maxM_);

    writeBinaryPOD(output, maxM0_);
    writeBinaryPOD(output, M_);
    writeBinaryPOD(output, ef_construction_);

    output.write(data_level0_memory_,
                 cur_element_count * size_data_per_element_);

    output.close();
  }

  void loadIndex(const std::string &location, SpaceInterface<dist_t> *s,
                 size_t max_elements_i = 0) {
    std::ifstream input(location, std::ios::binary);

    if (!input.is_open())
      throw std::runtime_error("Cannot open file");

    clear();
    // get file size:
    input.seekg(0, input.end);
    std::streampos total_filesize = input.tellg();
    input.seekg(0, input.beg);

    readBinaryPOD(input, offsetLevel0_);
    readBinaryPOD(input, max_elements_);
    readBinaryPOD(input, cur_element_count);

    size_t max_elements = max_elements_i;
    if (max_elements < cur_element_count)
      max_elements = max_elements_;
    max_elements_ = max_elements;
    readBinaryPOD(input, size_data_per_element_);
    readBinaryPOD(input, label_offset_);
    readBinaryPOD(input, offsetData_);
    readBinaryPOD(input, enterpoint_node_);
    readBinaryPOD(input, maxM_);

    readBinaryPOD(input, maxM0_);
    readBinaryPOD(input, M_);
    // double mult_;
    // readBinaryPOD(input, mult_);
    readBinaryPOD(input, ef_construction_);

    data_size_ = s->get_data_size();
    fstdistfunc_ = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();

    auto pos = input.tellg();

    input.seekg(cur_element_count * size_data_per_element_, input.cur);
    // throw exception if it either corrupted or old index
    if (input.tellg() != total_filesize)
      throw std::runtime_error("Index seems to be corrupted or unsupported");

    input.clear();
    /// Optional check end

    input.seekg(pos, input.beg);

    data_level0_memory_ = (char *)malloc(max_elements * size_data_per_element_);
    if (data_level0_memory_ == nullptr)
      throw std::runtime_error(
          "Not enough memory: loadIndex failed to allocate level0");
    input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

    size_links_per_element_ =
        maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);

    visited_list_pool_.reset(new VisitedListPool(1, max_elements));

    ef_ = 10;

    input.close();

    return;
  }

  unsigned short int getListCount(linklistsizeint *ptr) const {
    return *((unsigned short int *)ptr);
  }

  void setListCount(linklistsizeint *ptr, unsigned short int size) const {
    *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
  }

  /*
   * Adds point. Updates the point if it is already in the index.
   * If replacement of deleted elements is enabled: replaces previously deleted
   * point if any, updating it with new point
   */
  void addPoint(const void *data_point, labeltype label,
                bool replace_deleted = false) {

    addPoint(data_point, label, -1);
  }

  tableint addPoint(const void *data_point, labeltype label, int level) {
    tableint cur_c = cur_element_count++;

    tableint currObj = enterpoint_node_;

    memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_,
           0, size_data_per_element_);

    // Initialisation of the data and label
    memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
    memcpy(getDataByInternalId(cur_c), data_point, data_size_);

    if ((signed)currObj != -1) {
      std::priority_queue<std::pair<dist_t, tableint>,
                          std::vector<std::pair<dist_t, tableint>>,
                          CompareByFirst>
          top_candidates = searchBaseLayer(currObj, data_point);

      currObj =
          mutuallyConnectNewElement(data_point, cur_c, top_candidates);
    } else {
      // Do nothing for the first element
      enterpoint_node_ = 0;
    }

    return cur_c;
  }

  std::priority_queue<std::pair<dist_t, labeltype>>
  searchKnn(const void *query_data, size_t k,
            BaseFilterFunctor *isIdAllowed = nullptr) const {
    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count == 0)
      return result;

    tableint currObj = enterpoint_node_;
    dist_t curdist = fstdistfunc_(
        query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>,
                        CompareByFirst>
        top_candidates;

    top_candidates =
        searchBaseLayerST(currObj, query_data, std::max(ef_, k), isIdAllowed);

    while (top_candidates.size() > k) {
      top_candidates.pop();
    }
    while (top_candidates.size() > 0) {
      std::pair<dist_t, tableint> rez = top_candidates.top();
      result.push(std::pair<dist_t, labeltype>(rez.first, rez.second));
      top_candidates.pop();
    }
    return result;
  }
};
} // namespace hnswlib
