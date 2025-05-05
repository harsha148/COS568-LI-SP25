#pragma once

#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
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
        op_count_(0),
        insert_op_count_(0),
        min_flush_(10000),
        max_flush_(200000)
    {
        // start with min threshold
        flush_threshold_.store(min_flush_);
        stop_flag_.store(false);
        // persistent background flush thread
        flush_thread_ = std::thread(&HybridPGMLIPP::flushWorker, this);
    }

    ~HybridPGMLIPP() {
        // signal shutdown and wake the thread
        stop_flag_.store(true);
        flush_cv_.notify_one();
        if (flush_thread_.joinable())
            flush_thread_.join();
    }

    // Bulk-load into LIPP only
    uint64_t Build(const std::vector<KeyValue<KeyType>>& data,
                   size_t num_threads)
    {
        return lipp_index_.Build(data, num_threads);
    }

    // Lookup: skip PGM on pure misses, else PGM→LIPP fallback
    size_t EqualityLookup(const KeyType& key,
                          uint32_t thread_id) const
    {
        // count operation
        op_count_.fetch_add(1);

        // 1) If no pending inserts or key not in recent set ⇒ straight to LIPP
        {
            std::lock_guard<std::mutex> buf_lk(buffer_mutex_);
            if (insert_buffer_.empty() ||
                recent_keys_.count(key) == 0)
            {
                return lipp_index_.EqualityLookup(key, thread_id);
            }
            // 2) approximatePosition pre-filter (O(1) check)
            // auto ap = dp_index_.approximatePosition(key);
            // if (ap.lo >= ap.hi) {
            //     return lipp_index_.EqualityLookup(key, thread_id);
            // }
        }

        // 3) Otherwise do full PGM lookup (cheap, small range) then fallback
        size_t res = dp_index_.EqualityLookup(key, thread_id);
        return (res == util::OVERFLOW || res == util::NOT_FOUND)
               ? lipp_index_.EqualityLookup(key, thread_id)
               : res;
    }

    // RangeQuery: full PGM + LIPP (could also skip PGM, but we keep both)
    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi,
                        uint32_t thread_id) const
    {
        std::lock_guard<std::mutex> pgm_lk(pgm_mutex_);
        uint64_t r1 = dp_index_.RangeQuery(lo, hi, thread_id);
        uint64_t r2 = lipp_index_.RangeQuery(lo, hi, thread_id);
        return r1 + r2;
    }

    // Insert: buffer + PGM + track insert count + notify
    void Insert(const KeyValue<KeyType>& kv,
                uint32_t thread_id)
    {
        {
            std::lock_guard<std::mutex> buf_lk(buffer_mutex_);
            insert_buffer_.push_back(kv);
            recent_keys_.insert(kv.key);
        }

        {
            std::lock_guard<std::mutex> pgm_lk(pgm_mutex_);
            dp_index_.Insert(kv, thread_id);
        }

        insert_op_count_.fetch_add(1);

        // wake flush thread if we’ve filled the buffer
        if (insert_buffer_.size() >= flush_threshold_.load()) {
            flush_cv_.notify_one();
        }
    }

    std::string name() const { return "HybridPGMLIPP"; }
    std::vector<std::string> variants() const {
        return { SearchClass::name(), std::to_string(pgm_error) };
    }
    size_t size() const {
        std::lock_guard<std::mutex> pgm_lk(pgm_mutex_);
        return dp_index_.size() + lipp_index_.size();
    }
    bool applicable(bool unique, bool range_query, bool insert,
                    bool multithread, const std::string& ops_filename) const
    {
        return true;
    }

private:
    // Persistent background flush thread
    void flushWorker() {
        while (true) {
            std::vector<KeyValue<KeyType>> batch;
            {
                std::unique_lock<std::mutex> buf_lk(buffer_mutex_);
                flush_cv_.wait(buf_lk, [&] {
                    return stop_flag_.load() ||
                           insert_buffer_.size() >= flush_threshold_.load();
                });
                if (stop_flag_.load()) break;

                // grab the buffer
                batch.swap(insert_buffer_);
                recent_keys_.clear();
            }

            // 1) Bulk‐insert into LIPP
            for (auto &kv : batch) {
                lipp_index_.Insert(kv, 0);
            }

            // 2) Recompute adaptive threshold = min + (max-min)*(1 - insert_ratio)
            double r = 0;
            size_t ops = op_count_.load();
            if (ops) r = double(insert_op_count_.load()) / double(ops);
            size_t new_thr = size_t(min_flush_ +
                             (max_flush_ - min_flush_) * (1.0 - r));
            new_thr = std::clamp(new_thr, min_flush_, max_flush_);
            flush_threshold_.store(new_thr);
        }
    }

    // Underlying indexes
    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType>                               lipp_index_;

    // Protect PGM for concurrent calls (lookup and insert)
    mutable std::mutex pgm_mutex_;

    // Buffer + recent-key set
    mutable std::mutex                        buffer_mutex_;
    std::vector<KeyValue<KeyType>>            insert_buffer_;
    std::unordered_set<KeyType>               recent_keys_;

    // Adaptive flush threshold
    std::atomic<size_t>                       flush_threshold_;
    const size_t                              min_flush_;
    const size_t                              max_flush_;

    // Counters for adaptive ratio
    mutable std::atomic<size_t>                       op_count_;
    mutable std::atomic<size_t>                       insert_op_count_;

    // Flush‐thread
    std::thread                               flush_thread_;
    std::condition_variable                   flush_cv_;
    std::atomic<bool>                         stop_flag_;
};