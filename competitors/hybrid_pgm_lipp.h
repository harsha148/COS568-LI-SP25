#pragma once

#include "base.h"
#include "dynamic_pgm_index.h"   // must expose find_approximate_position()
#include "lipp.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <unordered_set>
#include <string>
#include <algorithm>

template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
  HybridPGMLIPP(const std::vector<int>& params)
    : dp_index_(params),
      lipp_index_(params),
      insert_count_(0),
      flush_threshold_(100000),
      total_ops_(0),
      insert_ops_(0),
      stop_flag_(false)
  {
    // start a persistent flush thread
    flush_thread_ = std::thread(&HybridPGMLIPP::flushWorker, this);
  }

  ~HybridPGMLIPP() {
    // shut down flush thread cleanly
    stop_flag_.store(true);
    flush_cv_.notify_one();
    if (flush_thread_.joinable())
      flush_thread_.join();
  }

  // initial build into LIPP only
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data,
                 size_t num_threads)
  {
    uint64_t bytes = lipp_index_.Build(data, num_threads);
    // reset counters and threshold
    total_ops_.store(0);
    insert_ops_.store(0);
    insert_count_ = 0;
    recent_keys_.clear();
    // threshold remains at default 100k
    return bytes;
  }

  size_t EqualityLookup(const KeyType& key,
                        uint32_t thread_id) const
  {
    total_ops_.fetch_add(1);

    // 1) If key not in recent buffer, skip PGM entirely
    {
      std::lock_guard lk(buffer_mutex_);
      if (!recent_keys_.count(key)) {
        return lipp_index_.EqualityLookup(key, thread_id);
      }
    }

    // 2) Prefilter via PGM approximate position
    auto ap = dp_index_.approximateposition(key);
    if (key < buffer_sorted_[ap.lo].key || key > buffer_sorted_[ap.hi].key) {
      // definitely not in PGM buffer
      return lipp_index_.EqualityLookup(key, thread_id);
    }

    // 3) Full PGM lookup, then fallback
    size_t res = dp_index_.EqualityLookup(key, thread_id);
    if (res != util::NOT_FOUND && res != util::OVERFLOW) {
      return res;
    }
    return lipp_index_.EqualityLookup(key, thread_id);
  }

  uint64_t RangeQuery(const KeyType& lo, const KeyType& hi,
                      uint32_t thread_id) const
  {
    total_ops_.fetch_add(1);

    // If nothing buffered, direct to LIPP
    {
      std::lock_guard lk(buffer_mutex_);
      if (insert_buffer_.empty()) {
        return lipp_index_.RangeQuery(lo, hi, thread_id);
      }
    }

    // Otherwise sum both
    uint64_t r1 = dp_index_.RangeQuery(lo, hi, thread_id);
    uint64_t r2 = lipp_index_.RangeQuery(lo, hi, thread_id);
    return r1 + r2;
  }

  void Insert(const KeyValue<KeyType>& kv,
              uint32_t thread_id)
  {
    total_ops_.fetch_add(1);
    insert_ops_.fetch_add(1);

    // buffer + recent‐keys
    {
      std::lock_guard lk(buffer_mutex_);
      insert_buffer_.push_back(kv);
      recent_keys_.insert(kv.key);
      ++insert_count_;
    }
    // always insert into PGM
    dp_index_.Insert(kv, thread_id);
  }

  std::string name() const { return "HybridPGMLIPP"; }
  std::vector<std::string> variants() const {
    return { SearchClass::name(), std::to_string(pgm_error) };
  }
  size_t size() const {
    // PGM + LIPP memory sizes
    return dp_index_.size() + lipp_index_.size();
  }
  bool applicable(bool u,bool r,bool i,bool m,const std::string&f) const {
    return true;
  }

private:
  void flushWorker() {
    while (true) {
      std::vector<KeyValue<KeyType>> batch;
      {
        std::unique_lock lk(buffer_mutex_);
        // wait for either data or shutdown
        flush_cv_.wait(lk, [&]{
          return stop_flag_.load() || insert_count_ >= flush_threshold_;
        });
        if (stop_flag_.load()) break;

        // grab buffer snapshot
        batch.swap(insert_buffer_);
        insert_count_ = 0;
        // rebuild sorted view for approximate‐position checks
        buffer_sorted_ = batch;
        std::sort(buffer_sorted_.begin(), buffer_sorted_.end(),
                  [](auto &a, auto &b){ return a.key < b.key; });
        // clear recent‐keys to bypass PGM on old entries
        recent_keys_.clear();
      }

      // 1) Bulk‐insert into LIPP (you could parallelize here if desired)
      for (auto &kv : batch) {
        lipp_index_.Insert(kv, /*thread=*/0);
      }

      // 2) Adapt flush threshold based on insert_ratio
      {
        double insert_ratio = 0;
        size_t tot = total_ops_.load();
        if (tot) insert_ratio = double(insert_ops_.load()) / tot;
        // linear‐interpolate threshold between min/max
        size_t min_t = 20'000, max_t = 500'000;
        size_t new_t = min_t + size_t((max_t - min_t) * (1.0 - insert_ratio));
        flush_threshold_ = std::clamp(new_t, min_t, max_t);
      }
    }
  }

  DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
  Lipp<KeyType>                               lipp_index_;

  // buffer + recent‐keys
  mutable std::mutex                         buffer_mutex_;
  std::vector<KeyValue<KeyType>>             insert_buffer_;
  std::vector<KeyValue<KeyType>>             buffer_sorted_;
  std::unordered_set<KeyType>                recent_keys_;
  size_t                                     insert_count_{0};

  // flush trigger & threshold
  std::atomic<size_t>                        flush_threshold_;
  std::thread                                flush_thread_;
  std::atomic<bool>                          stop_flag_;

  // for computing insert_ratio
  mutable std::atomic<size_t>                total_ops_;
  mutable std::atomic<size_t>                insert_ops_;
};