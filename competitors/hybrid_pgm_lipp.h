#pragma once

#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
  HybridPGMLIPP(const std::vector<int>& params)
    : dp_index_(params)
    , lipp_index_(params)
    , flush_threshold_(100000)
    , stop_flag_(false)
  {
    // start the background flusher
    flush_thread_ = std::thread(&HybridPGMLIPP::flushWorker, this);
  }

  ~HybridPGMLIPP() {
    {   // signal the flusher to exit
      std::lock_guard<std::mutex> lk(flush_mutex_);
      stop_flag_ = true;
      flush_cv_.notify_one();
    }
    if (flush_thread_.joinable())
      flush_thread_.join();
  }

  uint64_t Build(const std::vector<KeyValue<KeyType>>& data,
                 size_t num_threads) override
  {
    // just bulk‐load LIPP as before
    return lipp_index_.Build(data, num_threads);
  }

  size_t EqualityLookup(const KeyType& key,
                        uint32_t thread_id) const override
  {
    // If nothing is buffered, go straight to LIPP
    if (insert_count_.load(std::memory_order_relaxed) == 0) {
      return lipp_index_.EqualityLookup(key, thread_id);
    }

    // lock while touching either index
    std::lock_guard<std::mutex> lk(index_mutex_);

    // FIRST try PGM if the key is in our small buffer
    // (you can optimize with a recent_keys_ set here)
    size_t res = dp_index_.EqualityLookup(key, thread_id);
    if (res != util::NOT_FOUND && res != util::OVERFLOW)
      return res;

    // FALLBACK to LIPP
    return lipp_index_.EqualityLookup(key, thread_id);
  }

  uint64_t RangeQuery(const KeyType& lo,
                      const KeyType& hi,
                      uint32_t thread_id) const override
  {
    std::lock_guard<std::mutex> lk(index_mutex_);
    return dp_index_.RangeQuery(lo, hi, thread_id)
         + lipp_index_.RangeQuery(lo, hi, thread_id);
  }

  void Insert(const KeyValue<KeyType>& kv,
              uint32_t thread_id) override
  {
    { // buffer the new key/value
      std::lock_guard<std::mutex> buf_lk(buffer_mutex_);
      insert_buffer_.push_back(kv);
      ++insert_count_;
    }

    { // immediately push into the dynamic‐PGM (fast)
      std::lock_guard<std::mutex> lk(index_mutex_);
      dp_index_.Insert(kv, thread_id);
    }

    // if we've reached threshold, schedule a flush
    if (insert_count_.load(std::memory_order_relaxed) >= flush_threshold_) {
      std::lock_guard<std::mutex> lk(flush_mutex_);
      flush_cv_.notify_one();
    }
  }

  std::string name() const override {
    return "HybridPGMLIPP";
  }

  std::vector<std::string> variants() const override {
    return { SearchClass::name(), std::to_string(pgm_error) };
  }

  size_t size() const override {
    std::lock_guard<std::mutex> lk(index_mutex_);
    return dp_index_.size() + lipp_index_.size();
  }

private:
  void flushWorker() {
    while (true) {
      std::vector<KeyValue<KeyType>> batch;

      { // wait for work or shutdown
        std::unique_lock<std::mutex> lk(flush_mutex_);
        flush_cv_.wait(lk, [&] {
          return stop_flag_ 
              || insert_count_.load(std::memory_order_relaxed) >= flush_threshold_;
        });
        if (stop_flag_) break;

        // grab the pending buffer
        {
          std::lock_guard<std::mutex> buf_lk(buffer_mutex_);
          batch.swap(insert_buffer_);
          insert_count_.store(0, std::memory_order_relaxed);
        }
      }

      // now apply in one critical section:
      {
        std::lock_guard<std::mutex> lk(index_mutex_);

        // 1) move all batch items into LIPP in bulk
        for (auto &kv : batch)
          lipp_index_.Insert(kv, /*thread_id=*/0u);

        // 2) clear the PGM completely
        dp_index_.clear();
      }

      // optionally adjust flush_threshold_ here…
    }
  }

  // indexes (guarded by index_mutex_)
  DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
  Lipp<KeyType>                             lipp_index_;
  mutable std::mutex                        index_mutex_;

  // small buffer (guarded by buffer_mutex_)
  std::vector<KeyValue<KeyType>>            insert_buffer_;
  std::mutex                                buffer_mutex_;
  std::atomic<size_t>                       insert_count_;

  // background flush thread + queue trigger
  std::thread                               flush_thread_;
  std::mutex                                flush_mutex_;
  std::condition_variable                   flush_cv_;
  bool                                      stop_flag_;

  // how many buffered keys before flushing
  size_t                                    flush_threshold_;
};