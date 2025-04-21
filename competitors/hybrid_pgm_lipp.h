#ifndef TLI_HYBRID_PGM_LIPP_H
#define TLI_HYBRID_PGM_LIPP_H

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../util.h"
#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"

template <class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
 public:
  HybridPGMLIPP(const std::vector<int>& params) 
    : dpgm_(params), lipp_(params), flush_threshold_(0.05), lipp_element_count_(0) {}

  uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
    // Initially build LIPP with all data
    uint64_t build_time = lipp_.Build(data, num_threads);
    lipp_element_count_ = data.size();  // Track initial number of elements
    return build_time;
  }

  size_t EqualityLookup(const KeyType& lookup_key, uint32_t thread_id) const {
    // First check DPGM
    size_t dpgm_result = dpgm_.EqualityLookup(lookup_key, thread_id);
    if (dpgm_result != util::OVERFLOW) {
      return dpgm_result;
    }
    
    // If not found in DPGM, check LIPP
    return lipp_.EqualityLookup(lookup_key, thread_id);
  }

  uint64_t RangeQuery(const KeyType& lower_key, const KeyType& upper_key, uint32_t thread_id) const {
    // For range queries, we need to check both indexes
    uint64_t dpgm_result = dpgm_.RangeQuery(lower_key, upper_key, thread_id);
    uint64_t lipp_result = lipp_.RangeQuery(lower_key, upper_key, thread_id);
    return dpgm_result + lipp_result;
  }

  void Insert(const KeyValue<KeyType>& data, uint32_t thread_id) {
    // Always insert into DPGM first
    dpgm_.Insert(data, thread_id);
    dpgm_data.emplace_back(data);
    
    // Check if we need to flush to LIPP based on number of elements
    if (dpgm_data.size() >= flush_threshold_ * lipp_element_count_) {
      FlushToLIPP();
    }
  }

  std::string name() const { return "HybridPGMLIPP"; }

  std::size_t size() const { return dpgm_.size() + lipp_.size(); }

  bool applicable(bool unique, bool range_query, bool insert, bool multithread, const std::string& ops_filename) const {
    return !multithread;
  }

  std::vector<std::string> variants() const { 
    std::vector<std::string> vec;
    vec.push_back(SearchClass::name());
    vec.push_back(std::to_string(pgm_error));
    return vec;
  }

 private:
  void FlushToLIPP() {
    // Insert each key-value pair into LIPP
    for (const auto& kv : dpgm_data) {
      lipp_.Insert(kv, 0);  // Using thread_id 0 for simplicity
    }
    
    // Update LIPP element count
    lipp_element_count_ += dpgm_data.size();
    
    // Clear DPGM and create a new instance
    dpgm_ = DynamicPGM<KeyType, SearchClass, pgm_error>(std::vector<int>());
    dpgm_data.clear();
  }

  DynamicPGM<KeyType, SearchClass, pgm_error> dpgm_;
  Lipp<KeyType> lipp_; 
  std::vector<KeyValue<KeyType>> dpgm_data;
  double flush_threshold_;
  size_t lipp_element_count_;  // Track number of elements in LIPP
};

#endif // TLI_HYBRID_PGM_LIPP_H