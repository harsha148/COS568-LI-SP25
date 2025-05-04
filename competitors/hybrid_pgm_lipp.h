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
      flush_threshold_(0.05), 
      lipp_element_count_(0),
      is_flushing_(false),
      insert_count_(0),
      last_flush_time_(std::chrono::steady_clock::now()) {}

  ~HybridPGMLIPP() {
    if (flush_thread_.joinable()) {
      flush_thread_.join();
    }
  }

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
    // Insert into DPGM
    dpgm_.Insert(data, thread_id);
    
    // Add to buffer with lock
    {
      std::lock_guard<std::mutex> lock(buffer_mutex_);
      dpgm_data.emplace_back(data);
      insert_count_++;
    }
    
    // Check if we need to trigger a flush
    if (should_trigger_flush()) {
      trigger_flush();
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
  bool should_trigger_flush() {
    // Check if we're already flushing
    if (is_flushing_.load()) {
      return false;
    }

    // Check size-based threshold
    if (insert_count_.load() >= flush_threshold_ * lipp_element_count_) {
      return true;
    }

    // Check time-based threshold (flush every 100ms if we have data)
    auto now = std::chrono::steady_clock::now();
    if (insert_count_.load() > 0 && 
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_flush_time_).count() > 100) {
      return true;
    }

    return false;
  }

  void trigger_flush() {
    // Try to acquire flush lock
    bool expected = false;
    if (is_flushing_.compare_exchange_strong(expected, true)) {
      // Start async flush
      if (flush_thread_.joinable()) {
        flush_thread_.join();
      }
      flush_thread_ = std::thread(&HybridPGMLIPP::async_flush, this);
    }
  }

  void async_flush() {
    // Take snapshot of current data
    std::vector<KeyValue<KeyType>> snapshot;
    size_t count;
    {
      std::lock_guard<std::mutex> lock(buffer_mutex_);
      snapshot.swap(dpgm_data);
      count = insert_count_.load();
      insert_count_ = 0;
    }

    // Insert into LIPP
    for (const auto& kv : snapshot) {
      lipp_.Insert(kv, 0);
    }

    // Update LIPP element count
    lipp_element_count_ += count;

    // Clear DPGM and create new instance
    dpgm_ = DynamicPGM<KeyType, SearchClass, pgm_error>(std::vector<int>());

    // Update last flush time and release flush lock
    last_flush_time_ = std::chrono::steady_clock::now();
    is_flushing_.store(false);
  }

  DynamicPGM<KeyType, SearchClass, pgm_error> dpgm_;
  Lipp<KeyType> lipp_; 
  std::vector<KeyValue<KeyType>> dpgm_data;
  std::mutex buffer_mutex_;
  std::atomic<bool> is_flushing_;
  std::atomic<size_t> insert_count_;
  std::thread flush_thread_;
  double flush_threshold_;
  size_t lipp_element_count_;
  std::chrono::steady_clock::time_point last_flush_time_;
};

#endif // TLI_HYBRID_PGM_LIPP_H


