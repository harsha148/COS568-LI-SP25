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
#include <algorithm>

template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    HybridPGMLIPP(const std::vector<int>& params)
      : dp_index_(params),
        lipp_index_(params),
        insert_count_(0),
        flush_threshold_(0),
        initial_size_(0),
        lipp_key_count_(0),
        stop_flag_(false)
    {
        // Start the background flush thread
        flush_thread_ = std::thread(&HybridPGMLIPP::flushWorker, this);
    }

    ~HybridPGMLIPP() {
        // Signal and join flush thread
        stop_flag_.store(true);
        flush_cv_.notify_one();
        if (flush_thread_.joinable()) {
            flush_thread_.join();
        }
    }

    // Build: bulk-load into LIPP, then set static threshold = 5% of initial data size
    uint64_t Build(const std::vector<KeyValue<KeyType>>& data,
                   size_t num_threads)
    {
        uint64_t bytes = lipp_index_.Build(data, num_threads);
        initial_size_ = data.size();
        // Initialize LIPP key count and threshold
        lipp_key_count_.store(initial_size_);
        size_t threshold = (initial_size_ * 5 + 99) / 100;  // 5% rounded up
        if (threshold == 0) threshold = 1;
        flush_threshold_.store(threshold);
        return bytes;
    }

    // Equality lookup: try PGM for recent inserts, else LIPP
    size_t EqualityLookup(const KeyType& key,
                          uint32_t thread_id) const
    {
        std::lock_guard<std::mutex> lk(set_mutex_);
        if (inserted_set_.count(key)) {
            size_t res = dp_index_.EqualityLookup(key, thread_id);
            if (res != util::NOT_FOUND && res != util::OVERFLOW)
                return res;
        }
        return lipp_index_.EqualityLookup(key, thread_id);
    }

    // Range query: sum results from PGM and LIPP
    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi,
                        uint32_t thread_id) const
    {
        return dp_index_.RangeQuery(lo, hi, thread_id)
             + lipp_index_.RangeQuery(lo, hi, thread_id);
    }

    // Insert: buffer into PGM, record key, schedule flush when buffer ≥ threshold
    void Insert(const KeyValue<KeyType>& kv,
                uint32_t thread_id)
    {
        {
            std::lock_guard<std::mutex> lk(set_mutex_);
            inserted_set_.insert(kv.key);
        }
        dp_index_.Insert(kv, thread_id);

        {
            std::lock_guard<std::mutex> buf_lk(buffer_mutex_);
            insert_buffer_.push_back(kv);
            ++insert_count_;
        }
        if (insert_count_ >= flush_threshold_.load()) {
            std::vector<KeyValue<KeyType>> batch;
            {
                std::lock_guard<std::mutex> buf_lk(buffer_mutex_);
                batch.swap(insert_buffer_);
                insert_count_ = 0;
            }
            {
                std::lock_guard<std::mutex> q_lk(flush_mutex_);
                flush_queue_.push(std::move(batch));
            }
            flush_cv_.notify_one();
        }
    }

    // Metadata to satisfy Base interface
    std::string name() const { return "HybridPGMLIPP"; }
    std::vector<std::string> variants() const {
        return { SearchClass::name(), std::to_string(pgm_error) };
    }
    size_t size() const { return dp_index_.size() + lipp_index_.size(); }
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

            // Bulk-insert the batch into LIPP
            lipp_index_.BulkInsert(batch, /*thread_id=*/0u);

            // Update LIPP key count and recompute 5% threshold
            size_t newly_added = batch.size();
            size_t total_keys = lipp_key_count_.fetch_add(newly_added) + newly_added;
            size_t new_thresh  = (total_keys * 5 + 99) / 100;
            if (new_thresh == 0) new_thresh = 1;
            flush_threshold_.store(new_thresh);
        }
    }

    // Underlying PGM and LIPP indices
    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType>                               lipp_index_;

    // Fast-path set of recently inserted keys
    mutable std::mutex                   set_mutex_;
    std::unordered_set<KeyType>          inserted_set_;

    // Buffer and counters for pending LIPP migration
    std::mutex                                    buffer_mutex_;
    std::vector<KeyValue<KeyType>>               insert_buffer_;
    size_t                                        insert_count_;
    std::atomic<size_t>                           flush_threshold_;
    size_t                                        initial_size_;
    std::atomic<size_t>                           lipp_key_count_;

    // Flush-thread synchronization
    std::mutex                                    flush_mutex_;
    std::condition_variable                       flush_cv_;
    std::queue<std::vector<KeyValue<KeyType>>>    flush_queue_;
    std::thread                                   flush_thread_;
    std::atomic<bool>                             stop_flag_;
};