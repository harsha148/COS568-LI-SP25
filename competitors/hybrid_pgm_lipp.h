#ifndef TLI_HYBRID_PGM_LIPP_H
#define TLI_HYBRID_PGM_LIPP_H

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

#include "../util.h"
#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"

template <class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
 public:
  HybridPGMLIPP(const std::vector<int>& params) 
    : dpgm_(params), lipp_(params), 
      insert_count_(0),
      is_flushing_(false) {
    flush_threshold_ = 100000;  // Fixed threshold
  }

  ~HybridPGMLIPP() {
    if (flush_thread_.joinable()) {
      flush_thread_.join();
    }
  }

  uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
    return lipp_.Build(data, num_threads);
  }

  size_t EqualityLookup(const KeyType& lookup_key, uint32_t thread_id) const {
    // Always check DPGM first
    size_t dpgm_result = dpgm_.EqualityLookup(lookup_key, thread_id);
    if (dpgm_result != util::OVERFLOW && dpgm_result != util::NOT_FOUND) {
      return dpgm_result;
    }
    
    // If not found in DPGM, check LIPP
    return lipp_.EqualityLookup(lookup_key, thread_id);
  }

  uint64_t RangeQuery(const KeyType& lower_key, const KeyType& upper_key, uint32_t thread_id) const {
    // Always check both indexes
    uint64_t dpgm_result = dpgm_.RangeQuery(lower_key, upper_key, thread_id);
    uint64_t lipp_result = lipp_.RangeQuery(lower_key, upper_key, thread_id);
    return dpgm_result + lipp_result;
  }

  void Insert(const KeyValue<KeyType>& data, uint32_t thread_id) {
    // Always use DPGM and buffer for inserts
    {
      std::lock_guard<std::mutex> lock(buffer_mutex_);
      dpgm_data.emplace_back(data);
    }
    dpgm_.Insert(data, thread_id);
    insert_count_++;

    // Trigger flush if threshold reached
    if (insert_count_ >= flush_threshold_ && !is_flushing_.exchange(true)) {
      if (flush_thread_.joinable()) {
        flush_thread_.join();
      }
      flush_thread_ = std::thread(&HybridPGMLIPP::async_flush, this);
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
  void async_flush() {
    try {
      std::vector<KeyValue<KeyType>> snapshot;
      {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        snapshot.swap(dpgm_data);
        insert_count_ = 0;
      }

      // Insert into LIPP
      for (const auto& kv : snapshot) {
        lipp_.Insert(kv, 0);
      }

      // Clear DPGM and create new instance
      dpgm_ = DynamicPGM<KeyType, SearchClass, pgm_error>(std::vector<int>());
      
      is_flushing_.store(false);
    } catch (const std::exception& e) {
      std::cerr << "Error during flush: " << e.what() << std::endl;
      is_flushing_.store(false);
    }
  }

  DynamicPGM<KeyType, SearchClass, pgm_error> dpgm_;
  Lipp<KeyType> lipp_; 
  std::vector<KeyValue<KeyType>> dpgm_data;
  std::mutex buffer_mutex_;
  std::atomic<bool> is_flushing_;
  std::atomic<size_t> insert_count_;
  std::thread flush_thread_;
  size_t flush_threshold_;
};

#endif // TLI_HYBRID_PGM_LIPP_H


