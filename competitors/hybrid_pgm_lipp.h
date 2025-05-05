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
      : params_(params),
        dp_index_(params),
        lipp_index_(params),
        insert_count_(0),
        flush_threshold_(0),
        lipp_key_count_(0),
        stop_flag_(false)
    {
        // Launch background flush thread
        flush_thread_ = std::thread(&HybridPGMLIPP::flushWorker, this);
    }

    ~HybridPGMLIPP() {
        // Signal shutdown and join
        stop_flag_.store(true);
        flush_cv_.notify_one();
        if (flush_thread_.joinable())
            flush_thread_.join();
    }

    // Build: bulk-load into LIPP, then set flush_threshold = 5% of initial key-count
    uint64_t Build(const std::vector<KeyValue<KeyType>>& data,
                   size_t num_threads)
    {
        uint64_t bytes = lipp_index_.Build(data, num_threads);
        size_t initial = data.size();
        lipp_key_count_.store(initial);
        size_t threshold = (initial * 5 + 99) / 100;  // 5% rounded up
        flush_threshold_.store(std::max<size_t>(1, threshold));
        return bytes;
    }

    // EqualityLookup: if key is in 'recent' set, check PGM; else go straight to LIPP
    size_t EqualityLookup(const KeyType& key,
                          uint32_t thread_id) const
    {
        // 1) Fast‑path membership check
        {
            std::lock_guard<std::mutex> set_lk(set_mutex_);
            if (inserted_set_.count(key)) {
                // 2) PGM lookup under lock
                std::lock_guard<std::mutex> pgm_lk(pgm_mutex_);
                size_t res = dp_index_.EqualityLookup(key, thread_id);
                if (res != util::NOT_FOUND && res != util::OVERFLOW)
                    return res;
            }
        }
        // 3) Fallback to LIPP
        return lipp_index_.EqualityLookup(key, thread_id);
    }

    // RangeQuery: sum results from both
    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi,
                        uint32_t thread_id) const
    {
        // PGM under lock, LIPP directly
        std::lock_guard<std::mutex> pgm_lk(pgm_mutex_);
        uint64_t r1 = dp_index_.RangeQuery(lo, hi, thread_id);
        uint64_t r2 = lipp_index_.RangeQuery(lo, hi, thread_id);
        return r1 + r2;
    }

    // Insert: record key, insert into PGM, buffer for LIPP
    void Insert(const KeyValue<KeyType>& kv,
                uint32_t thread_id)
    {
        {
            std::lock_guard<std::mutex> set_lk(set_mutex_);
            inserted_set_.insert(kv.key);
        }
        {
            std::lock_guard<std::mutex> pgm_lk(pgm_mutex_);
            dp_index_.Insert(kv, thread_id);
        }

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

    // Metadata
    std::string name() const { return "HybridPGMLIPP"; }
    std::vector<std::string> variants() const {
        return { SearchClass::name(), std::to_string(pgm_error) };
    }
    size_t size() const { 
        // sum PGM + LIPP memory sizes
        std::lock_guard<std::mutex> pgm_lk(pgm_mutex_);
        return dp_index_.size() + lipp_index_.size(); 
    }
    bool applicable(bool u,bool r,bool i,bool m,const std::string&f) const {
        return true;
    }

private:
    // Background flush thread: drain queue → LIPP → clear PGM
    void flushWorker() {
        while (true) {
            std::vector<KeyValue<KeyType>> batch;
            {
                std::unique_lock<std::mutex> lk(flush_mutex_);
                flush_cv_.wait(lk, [&]{
                    return stop_flag_.load() || !flush_queue_.empty();
                });
                if (stop_flag_.load() && flush_queue_.empty())
                    break;
                batch = std::move(flush_queue_.front());
                flush_queue_.pop();
            }

            // 1) Bulk‐insert into LIPP
            lipp_index_.BulkInsert(batch, /*thread_id=*/0u);

            // 2) Account for new keys & recompute threshold
            size_t added = batch.size();
            size_t total = lipp_key_count_.fetch_add(added) + added;
            size_t thresh = (total * 5 + 99) / 100;
            flush_threshold_.store(std::max<size_t>(1, thresh));

            // 3) Clear the PGM index entirely (so only unflushed keys remain in PGM)
            {
                std::lock_guard<std::mutex> pgm_lk(pgm_mutex_);
                dp_index_ = DynamicPGM<KeyType, SearchClass, pgm_error>(params_);
            }
        }
    }

    // Constructor‐saved params so we can reinitialize PGM after each flush
    const std::vector<int>                 params_;
    DynamicPGM<KeyType,SearchClass,pgm_error> dp_index_;
    Lipp<KeyType>                          lipp_index_;

    // Protects dp_index_ on concurrent PGM operations
    mutable std::mutex                     pgm_mutex_;

    // Fast‐path membership test
    mutable std::mutex                     set_mutex_;
    std::unordered_set<KeyType>            inserted_set_;

    // Buffer + threshold
    std::mutex                             buffer_mutex_;
    std::vector<KeyValue<KeyType>>        insert_buffer_;
    size_t                                 insert_count_;
    std::atomic<size_t>                    flush_threshold_;

    // Track total keys in LIPP
    std::atomic<size_t>                    lipp_key_count_;

    // Flush‐thread queue + sync
    std::mutex                             flush_mutex_;
    std::condition_variable                flush_cv_;
    std::queue<std::vector<KeyValue<KeyType>>> flush_queue_;
    std::thread                            flush_thread_;
    std::atomic<bool>                      stop_flag_;
};