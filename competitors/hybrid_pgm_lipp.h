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

template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    // Constructor: initialize both indexes and launch the background flush thread
    HybridPGMLIPP(const std::vector<int>& params)
      : dp_index_(params),
        lipp_index_(params),
        insert_count_(0),
        stop_flag_(false),
        flush_threshold_(100000)
    {
        flush_thread_ = std::thread(&HybridPGMLIPP::flushWorker, this);
    }

    // Destructor: signal the worker to stop, wake it up, and join
    ~HybridPGMLIPP() {
        stop_flag_.store(true);
        flush_cv_.notify_one();
        if (flush_thread_.joinable()) {
            flush_thread_.join();
        }
    }

    // Bulk‐load phase: we delegate entirely to LIPP for build
    uint64_t Build(const std::vector<KeyValue<KeyType>>& data,
                   size_t num_threads)
    {
        return lipp_index_.Build(data, num_threads);
    }

    // Equality lookup: first try PGM, if not found or overflow, fall back to LIPP
    size_t EqualityLookup(const KeyType& key, uint32_t thread_id) const override {
        size_t res = dp_index_.EqualityLookup(key, thread_id);
        if (res == util::OVERFLOW || res == util::NOT_FOUND) {
            return lipp_index_.EqualityLookup(key, thread_id);
        }
        return res;
    }

    // Range query: sum results from PGM and from LIPP
    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi,
                        uint32_t thread_id) const override
    {
        return dp_index_.RangeQuery(lo, hi, thread_id)
             + lipp_index_.RangeQuery(lo, hi, thread_id);
    }

    // Insert: buffer into PGM, schedule batch flush when threshold reached
    void Insert(const KeyValue<KeyType>& kv, uint32_t thread_id) override {
        // 1) Buffer the new key/value for later LIPP insertion
        {
            std::lock_guard<std::mutex> buf_lock(buffer_mutex_);
            insert_buffer_.push_back(kv);
            ++insert_count_;
        }

        // 2) Immediately insert into the PGM index
        dp_index_.Insert(kv, thread_id);

        // 3) If we've accumulated enough, snapshot & enqueue for background flush
        if (insert_count_ >= flush_threshold_) {
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

    // Metadata
    std::string name() const override {
        return "HybridPGMLIPP";
    }
    std::vector<std::string> variants() const override {
        return { SearchClass::name(), std::to_string(pgm_error) };
    }
    size_t size() const override {
        return dp_index_.size() + lipp_index_.size();
    }

    // Use this to filter out unsuitable workloads (here we accept all)
    bool applicable(bool unique, bool range_query, bool insert,
                    bool multithread, const std::string& ops_filename) const override
    {
        return true;
    }

private:
    // Background worker: pop batches and do bulk‐inserts into LIPP
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
            // Bulk‐insert into LIPP (thread_id = 0 for simplicity)
            for (auto &kv : batch) {
                lipp_index_.Insert(kv, 0);
            }
        }
    }

    // Underlying indexes
    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType>                         lipp_index_;

    // Buffering pending inserts for LIPP
    std::vector<KeyValue<KeyType>>       insert_buffer_;
    size_t                               insert_count_;
    size_t                               flush_threshold_;
    std::mutex                           buffer_mutex_;

    // Flush‐queue and worker synchronization
    std::thread                          flush_thread_;
    std::mutex                           flush_mutex_;
    std::condition_variable              flush_cv_;
    std::queue<std::vector<KeyValue<KeyType>>> flush_queue_;
    std::atomic<bool>                    stop_flag_;
};