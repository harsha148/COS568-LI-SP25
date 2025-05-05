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
    // Constructor: initialize both indexes and launch background flush thread
    HybridPGMLIPP(const std::vector<int>& params)
      : dp_index_(params),
        lipp_index_(params),
        insert_count_(0),
        flush_threshold_(100000),
        stop_flag_(false)
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

    // Bulk‐load: delegate entirely to LIPP
    uint64_t Build(const std::vector<KeyValue<KeyType>>& data,
                   size_t num_threads)
    {
        return lipp_index_.Build(data, num_threads);
    }

    // Equality lookup: first try PGM, if not found/overflow, fall back to LIPP
    size_t EqualityLookup(const KeyType& key,
                          uint32_t thread_id) const
    {
        size_t res = dp_index_.EqualityLookup(key, thread_id);
        if (res == util::OVERFLOW || res == util::NOT_FOUND) {
            return lipp_index_.EqualityLookup(key, thread_id);
        }
        return res;
    }

    // Range query: sum results from PGM and LIPP
    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi,
                        uint32_t thread_id) const
    {
        return dp_index_.RangeQuery(lo, hi, thread_id)
             + lipp_index_.RangeQuery(lo, hi, thread_id);
    }

    // Insert: buffer for LIPP, immediately insert into PGM,
    // then enqueue a batch when threshold is reached
    void Insert(const KeyValue<KeyType>& kv,
                uint32_t thread_id)
    {
        // 1) Buffer for later LIPP insertion
        {
            std::lock_guard<std::mutex> buf_lock(buffer_mutex_);
            insert_buffer_.push_back(kv);
            ++insert_count_;
        }

        // 2) Immediate PGM insert
        dp_index_.Insert(kv, thread_id);

        // 3) Flush if we've buffered enough
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

    // Metadata to satisfy Competitor interface
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
    // Background flush worker
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
            // Bulk‐insert into LIPP (thread‑id 0 for simplicity)
            for (auto &kv : batch) {
                lipp_index_.Insert(kv, /*thread_id=*/0u);
            }
        }
    }

    // Underlying indexes
    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType>                              lipp_index_;

    // Buffering for pending LIPP inserts
    std::vector<KeyValue<KeyType>>           insert_buffer_;
    size_t                                   insert_count_;
    size_t                                   flush_threshold_;
    std::mutex                               buffer_mutex_;

    // Flush‑queue and synchronization
    std::thread                              flush_thread_;
    std::mutex                               flush_mutex_;
    std::condition_variable                  flush_cv_;
    std::queue<std::vector<KeyValue<KeyType>>> flush_queue_;
    std::atomic<bool>                        stop_flag_;
};