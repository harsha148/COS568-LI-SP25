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
#include <string>
#include <chrono>
#include <algorithm>

template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    // Constructor: initialize indexes & tuning params, then launch flush thread
    HybridPGMLIPP(const std::vector<int>& params)
      : dp_index_(params),
        lipp_index_(params),
        insert_count_(0),
        // initial static threshold
        flush_threshold_(100000),
        // EWMA starts assuming ~10ms per flush
        avg_flush_ms_(10.0),
        target_flush_ms_(10.0),
        alpha_(0.2),
        min_threshold_(10000),
        max_threshold_(500000),
        stop_flag_(false)
    {
        flush_thread_ = std::thread(&HybridPGMLIPP::flushWorker, this);
    }

    // Destructor: tell worker to stop, wake it, join it
    ~HybridPGMLIPP() {
        stop_flag_.store(true);
        flush_cv_.notify_one();
        if (flush_thread_.joinable()) {
            flush_thread_.join();
        }
    }

    // Bulk‑load: use LIPP’s Build
    uint64_t Build(const std::vector<KeyValue<KeyType>>& data,
                   size_t num_threads)
    {
        return lipp_index_.Build(data, num_threads);
    }

    // EqualityLookup: PGM first, then LIPP on overflow/not‑found
    size_t EqualityLookup(const KeyType& key,
                          uint32_t thread_id) const
    {
        size_t res = dp_index_.EqualityLookup(key, thread_id);
        if (res == util::OVERFLOW || res == util::NOT_FOUND) {
            return lipp_index_.EqualityLookup(key, thread_id);
        }
        return res;
    }

    // RangeQuery: accumulate results from both indexes
    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi,
                        uint32_t thread_id) const
    {
        return dp_index_.RangeQuery(lo, hi, thread_id)
             + lipp_index_.RangeQuery(lo, hi, thread_id);
    }

    // Insert: buffer for LIPP, immediate PGM insert, then schedule flush if needed
    void Insert(const KeyValue<KeyType>& kv,
                uint32_t thread_id)
    {
        {
            std::lock_guard<std::mutex> buf_lock(buffer_mutex_);
            insert_buffer_.push_back(kv);
            ++insert_count_;
        }

        dp_index_.Insert(kv, thread_id);

        // Check against the *dynamic* threshold
        if (insert_count_ >= flush_threshold_.load()) {
            // snapshot & reset buffer
            std::vector<KeyValue<KeyType>> batch;
            {
                std::lock_guard<std::mutex> buf_lock(buffer_mutex_);
                batch.swap(insert_buffer_);
                insert_count_ = 0;
            }
            // enqueue for background flush
            {
                std::lock_guard<std::mutex> queue_lock(flush_mutex_);
                flush_queue_.push(std::move(batch));
            }
            flush_cv_.notify_one();
        }
    }

    // Metadata
    std::string name() const {
        return "HybridPGMLIPP";
    }
    std::vector<std::string> variants() const {
        return { SearchClass::name(), std::to_string(pgm_error) };
    }
    size_t size() const {
        return dp_index_.size() + lipp_index_.size();
    }
    bool applicable(bool unique, bool range_query, bool insert,
                    bool multithread, const std::string& ops_filename) const
    {
        return true;
    }

private:
    // Worker thread: drains flush_queue_, measures time, adjusts threshold
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

            // Time this bulk flush
            auto t0 = std::chrono::steady_clock::now();
            for (auto &kv : batch) {
                lipp_index_.Insert(kv, /*thread_id=*/0u);
            }
            auto t1 = std::chrono::steady_clock::now();
            double dur_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Update EWMA of flush time
            double prev = avg_flush_ms_.load();
            double next = alpha_ * dur_ms + (1.0 - alpha_) * prev;
            avg_flush_ms_.store(next);

            // Adjust threshold toward target_flush_ms_
            double ratio = target_flush_ms_ / next;
            size_t new_thresh = static_cast<size_t>(flush_threshold_.load() * ratio);
            new_thresh = std::clamp(new_thresh, min_threshold_, max_threshold_);
            flush_threshold_.store(new_thresh);
        }
    }

    // Underlying indexes
    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType>                               lipp_index_;

    // Buffer for pending LIPP inserts
    std::vector<KeyValue<KeyType>>       insert_buffer_;
    size_t                               insert_count_;
    std::mutex                           buffer_mutex_;

    // Adaptive threshold & EWMA state
    std::atomic<size_t>                  flush_threshold_;
    std::atomic<double>                  avg_flush_ms_;
    double                               target_flush_ms_;
    double                               alpha_;
    size_t                               min_threshold_;
    size_t                               max_threshold_;

    // Flush‑queue + worker sync
    std::thread                          flush_thread_;
    std::mutex                           flush_mutex_;
    std::condition_variable              flush_cv_;
    std::queue<std::vector<KeyValue<KeyType>>> flush_queue_;
    std::atomic<bool>                    stop_flag_;
};