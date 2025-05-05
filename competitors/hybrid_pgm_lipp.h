#pragma once

#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <unordered_set>
#include <string>
#include <chrono>
#include <algorithm>

template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    HybridPGMLIPP(const std::vector<int>& params)
      : dp_index_(params),
        lipp_index_(params),
        insert_count_(0),
        flush_threshold_(100000),
        avg_flush_ms_(10.0),
        // We will compute target dynamically between these:
        low_target_flush_ms_(5.0),
        high_target_flush_ms_(50.0),
        alpha_(0.2),
        min_threshold_(10000),
        max_threshold_(500000),
        op_count_(0),
        insert_op_count_(0),
        stop_flag_(false)
    {
        flush_thread_ = std::thread(&HybridPGMLIPP::flushWorker, this);
    }

    ~HybridPGMLIPP() {
        stop_flag_.store(true);
        flush_cv_.notify_one();
        if (flush_thread_.joinable()) {
            flush_thread_.join();
        }
    }

    uint64_t Build(const std::vector<KeyValue<KeyType>>& data,
                   size_t num_threads)
    {
        return lipp_index_.Build(data, num_threads);
    }

    size_t EqualityLookup(const KeyType& key,
                          uint32_t thread_id) const
    {
        // Track total operations
        op_count_.fetch_add(1);

        // If recently inserted, try PGM first
        {
            std::lock_guard<std::mutex> lk(set_mutex_);
            if (inserted_set_.count(key)) {
                size_t res = dp_index_.EqualityLookup(key, thread_id);
                if (res != util::NOT_FOUND && res != util::OVERFLOW)
                    return res;
            }
        }
        // Otherwise (or on PGM miss), go to LIPP directly
        return lipp_index_.EqualityLookup(key, thread_id);
    }

    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi,
                        uint32_t thread_id) const
    {
        // Still sum both, as range queries span both indexes
        return dp_index_.RangeQuery(lo, hi, thread_id)
             + lipp_index_.RangeQuery(lo, hi, thread_id);
    }

    void Insert(const KeyValue<KeyType>& kv,
                uint32_t thread_id)
    {
        // Track total and insert ops
        op_count_.fetch_add(1);
        insert_op_count_.fetch_add(1);

        // Remember this key for fast-path lookup
        {
            std::lock_guard<std::mutex> lk(set_mutex_);
            inserted_set_.insert(kv.key);
        }

        // Always insert into the PGM index immediately
        dp_index_.Insert(kv, thread_id);

        // Buffer for later LIPP migration
        {
            std::lock_guard<std::mutex> buf_lock(buffer_mutex_);
            insert_buffer_.push_back(kv);
            ++insert_count_;
        }

        // If we’ve buffered enough, snapshot & schedule a background flush
        if (insert_count_ >= flush_threshold_.load()) {
            std::vector<KeyValue<KeyType>> batch;
            {
                std::lock_guard<std::mutex> buf_lock(buffer_mutex_);
                batch.swap(insert_buffer_);
                insert_count_ = 0;
            }
            {
                std::lock_guard<std::mutex> queue_lock(flush_mutex_);
                flush_queue_.push(std::move(batch));
            }
            flush_cv_.notify_one();
        }
    }

    std::string name() const override { return "HybridPGMLIPP"; }
    std::vector<std::string> variants() const override {
        return { SearchClass::name(), std::to_string(pgm_error) };
    }
    size_t size() const override {
        return dp_index_.size() + lipp_index_.size();
    }
    bool applicable(bool unique, bool range_query, bool insert,
                    bool multithread, const std::string& ops_filename) const override
    {
        return true;
    }

private:
    // Background flush thread
    void flushWorker() {
        while (true) {
            std::vector<KeyValue<KeyType>> batch;
            {
                std::unique_lock<std::mutex> lk(flush_mutex_);
                flush_cv_.wait(lk, [&] {
                    return stop_flag_.load() || !flush_queue_.empty();
                });
                if (stop_flag_.load() && flush_queue_.empty())
                    break;
                batch = std::move(flush_queue_.front());
                flush_queue_.pop();
            }

            // Time the bulk‐insert
            auto t0 = std::chrono::steady_clock::now();
            lipp_index_.BulkInsert(batch, 0u);
            auto t1 = std::chrono::steady_clock::now();
            double dur_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Update EWMA of flush latency
            double prev = avg_flush_ms_.load();
            double next = alpha_*dur_ms + (1.0 - alpha_)*prev;
            avg_flush_ms_.store(next);

            // Compute observed insert ratio
            double ratio = 0.0;
            size_t ops = op_count_.load();
            if (ops) ratio = double(insert_op_count_.load()) / double(ops);

            // Interpolate target flush time
            double target = low_target_flush_ms_ +
                            (high_target_flush_ms_ - low_target_flush_ms_) * ratio;

            // Adjust threshold: bigger if heavy‑insert, smaller if heavy‑lookup
            double scale = target / next;
            size_t new_thresh = static_cast<size_t>(flush_threshold_.load() * scale);
            new_thresh = std::clamp(new_thresh, min_threshold_, max_threshold_);
            flush_threshold_.store(new_thresh);
        }
    }

    // Underlying indexes
    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType>                               lipp_index_;

    // Fast‐path lookup for recently inserted keys
    mutable std::mutex                   set_mutex_;
    std::unordered_set<KeyType>          inserted_set_;

    // Buffer for pending LIPP inserts
    std::mutex                           buffer_mutex_;
    std::vector<KeyValue<KeyType>>       insert_buffer_;
    size_t                               insert_count_;

    // **NEW** counters you forgot to declare:
    mutable std::atomic<size_t> op_count_{0};
    mutable std::atomic<size_t> insert_op_count_{0};

    // Adaptive threshold & EWMA state
    std::atomic<size_t>                  flush_threshold_;
    std::atomic<double>                  avg_flush_ms_;
    double                               low_target_flush_ms_, high_target_flush_ms_;
    double                               alpha_;
    size_t                               min_threshold_, max_threshold_;

    // Background‐flush thread & queue
    std::thread                          flush_thread_;
    std::mutex                           flush_mutex_;
    std::condition_variable              flush_cv_;
    std::queue<std::vector<KeyValue<KeyType>>> flush_queue_;
    std::atomic<bool>                    stop_flag_;
};