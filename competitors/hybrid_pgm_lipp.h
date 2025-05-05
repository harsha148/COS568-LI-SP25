#pragma once

#include "base.h"
#include "dynamic_pgm_index.h"  // must expose find_approximate_position()
#include "lipp.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <unordered_set>
#include <algorithm>

template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    HybridPGMLIPP(const std::vector<int>& params)
      : dp_index_(params)
      , lipp_index_(params)
      , insert_count_(0)
      , flush_threshold_(100000)
      , flushing_(false)
      , total_ops_(0)
      , insert_ops_(0)
    {}

    ~HybridPGMLIPP() {
        if (flush_thread_.joinable())
            flush_thread_.join();
    }

    uint64_t Build(const std::vector<KeyValue<KeyType>>& data,
                   size_t num_threads)
    {
        // Only build LIPP initially
        return lipp_index_.Build(data, num_threads);
    }

    size_t EqualityLookup(const KeyType& key,
                          uint32_t thread_id) const
    {
        total_ops_.fetch_add(1);
        // 2) Bypass PGM if key not in recent_keys_
        {
            std::lock_guard<std::mutex> lk(buffer_mutex_);
            if (!recent_keys_.count(key)) {
                return lipp_index_.EqualityLookup(key, thread_id);
            }
        }

        // 3) Prefilter with approximatePosition
        auto ap = dp_index_.find_approximate_position(key);
        {
            std::lock_guard<std::mutex> lk(buffer_mutex_);
            if (buffer_sorted_.empty() ||
                key < buffer_sorted_[ap.lo].key ||
                key > buffer_sorted_[ap.hi].key)
            {
                return lipp_index_.EqualityLookup(key, thread_id);
            }
        }

        // 4) Full PGM lookup, then fallback
        size_t res = dp_index_.EqualityLookup(key, thread_id);
        if (res != util::NOT_FOUND && res != util::OVERFLOW)
            return res;
        return lipp_index_.EqualityLookup(key, thread_id);
    }

    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi,
                        uint32_t thread_id) const
    {
        total_ops_.fetch_add(1);

        // 3) Otherwise sum both
        uint64_t r1 = dp_index_.RangeQuery(lo, hi, thread_id);
        uint64_t r2 = lipp_index_.RangeQuery(lo, hi, thread_id);
        return r1 + r2;
    }

    void Insert(const KeyValue<KeyType>& kv,
                uint32_t thread_id)
    {
        total_ops_.fetch_add(1);
        insert_ops_.fetch_add(1);

        {
            std::lock_guard<std::mutex> lk(buffer_mutex_);
            insert_buffer_.push_back(kv);
            recent_keys_.insert(kv.key);
            ++insert_count_;
        }
        // Always insert into PGM
        dp_index_.Insert(kv, thread_id);
        if (insert_count_ >= flush_threshold_ && !flushing_.exchange(true)) {
          if (flush_thread_.joinable()) flush_thread_.join();
          flush_thread_ = std::thread(&HybridPGMLIPP::flush_to_lipp, this);
        }

    }

    std::string name() const  { return "HybridPGMLIPP"; }
    std::vector<std::string> variants() const {
        return { SearchClass::name(), std::to_string(pgm_error) };
    }
    size_t size() const  {
        return dp_index_.size() + lipp_index_.size();
    }
    bool applicable(bool unique, bool range_query, bool insert,
                    bool multithread, const std::string& ops_filename) const 
    {
        return !multithread;
    }

private:
    // One‐shot flush into LIPP (spawned on demand)
    void flush_to_lipp() {
        // Snapshot + clear buffer
        std::vector<KeyValue<KeyType>> batch;
        {
            std::lock_guard<std::mutex> lk(buffer_mutex_);
            batch.swap(insert_buffer_);
            insert_count_ = 0;
            // Prepare sorted copy for approximatePosition prefilter
            buffer_sorted_ = batch;
            std::sort(buffer_sorted_.begin(), buffer_sorted_.end(),
                      [](auto &a, auto &b){ return a.key < b.key; });
            recent_keys_.clear();
        }

        // Bulk‐insert into LIPP (could parallelize here)
        for (auto &kv : batch) {
            lipp_index_.Insert(kv, /*thread_id=*/0u);
        }

        // Adaptive threshold: inversely proportional to insert_ratio
        double ratio = 0;
        {
            double tot = double(total_ops_.load());
            if (tot > 0) ratio = double(insert_ops_.load()) / tot;
        }
        // When insert‑heavy (ratio→1) use small threshold; when lookup‑heavy (ratio→0) use large
        size_t min_t = 20'000, max_t = 500'000;
        size_t t = min_t + size_t((max_t - min_t) * (1.0 - ratio));
        flush_threshold_ = std::clamp(t, min_t, max_t);

        // Allow next flush
        flushing_.store(false);
    }

    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType>                               lipp_index_;

    // Buffer + recent‑keys + sorted buffer
    mutable std::mutex                         buffer_mutex_;
    std::vector<KeyValue<KeyType>>             insert_buffer_;
    std::vector<KeyValue<KeyType>>             buffer_sorted_;
    std::unordered_set<KeyType>                recent_keys_;
    size_t                                     insert_count_;

    // Flush threading
    size_t                                     flush_threshold_;
    std::atomic<bool>                          flushing_;
    std::thread                                flush_thread_;

    // Counters for adaptive threshold
    mutable std::atomic<size_t>                total_ops_;
    mutable std::atomic<size_t>                insert_ops_;
};