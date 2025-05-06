#pragma once

#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <algorithm>

template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    HybridPGMLIPP(const std::vector<int>& params)
      : dp_index_(params),
        lipp_index_(params),
        insert_count_(0),
        total_insert_count_(0),
        total_ops_count(0),
        flushing_(false),
        flush_threshold_(1000000)
    {}

    ~HybridPGMLIPP() {
        if (flush_thread_.joinable()) flush_thread_.join();
    }

    // Build only LIPP initially
    uint64_t Build(const std::vector<KeyValue<KeyType>>& data,
                   size_t num_threads)
    {
        // build LIPP and record base size
        uint64_t t = lipp_index_.Build(data, num_threads);
        total_insert_count_ = data.size();
        total_ops_count = data.size();
        return t;
    }

    // Hybrid lookup: fast path into LIPP when buffer small, else PGM→LIPP
    size_t EqualityLookup(const KeyType& key,
                          uint32_t thread_id) const
    {
        total_ops_count++;
        std::lock_guard<std::mutex> lk(index_mutex_);

        // purely read‐heavy? hit LIPP first
        if (total_insert_count_/total_ops_count < 0.5) {
            auto v = lipp_index_.EqualityLookup(key, thread_id);
            if (v != util::NOT_FOUND) return v;
            return dp_index_.EqualityLookup(key, thread_id);
        }

        // normal: try PGM then LIPP
        auto res = dp_index_.EqualityLookup(key, thread_id);
        if (res != util::NOT_FOUND && res != util::OVERFLOW)
            return res;
        return lipp_index_.EqualityLookup(key, thread_id);
    }

    // RangeQuery sums both under lock
    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi,
                        uint32_t thread_id) const
    {
        total_ops_count++;
        std::lock_guard<std::mutex> lk(index_mutex_);
        return dp_index_.RangeQuery(lo, hi, thread_id)
             + lipp_index_.RangeQuery(lo, hi, thread_id);
    }

    // Insert into PGM + buffer; when threshold reached, fire background flush
    void Insert(const KeyValue<KeyType>& kv, uint32_t thread_id)
    {
        {   // buffer the key for later flush
            std::lock_guard<std::mutex> buf_lk(buffer_mutex_);
            insert_buffer_.push_back(kv);
            insert_count_++;
            total_insert_count_++;
            total_ops_count++;
        }

        {   // PGM insert under index lock
            std::lock_guard<std::mutex> lk(index_mutex_);
            dp_index_.Insert(kv, thread_id);
        }

        // dynamic threshold ~5% of LIPP size
        size_t dyn_thresh = std::max(flush_threshold_, total_insert_count_/20);

        if (insert_count_ >= dyn_thresh && !flushing_.exchange(true)) {
            dp_index_ = DynamicPGM<KeyType, SearchClass, pgm_error>(std::vector<int>());
            // join previous flush if any
            if (flush_thread_.joinable()) flush_thread_.join();
            // launch new flush
            flush_thread_ = std::thread(&HybridPGMLIPP::flush_to_lipp, this);
        }
    }

    std::string name()    const { return "HybridPGMLIPP"; }
    std::vector<std::string> variants() const {
        return { SearchClass::name(), std::to_string(pgm_error) };
    }
    size_t size() const {
        std::lock_guard<std::mutex> lk(index_mutex_);
        return dp_index_.size() + lipp_index_.size();
    }
    bool applicable(bool unique, bool range_query, bool insert,
                    bool multithread, const std::string&) const
    {
        return !multithread;
    }

private:
    // drain buffer → LIPP, then clear PGM; all under index lock
    void flush_to_lipp() {
        // 1) snapshot & clear buffer
        std::vector<KeyValue<KeyType>> batch;
        {
            std::lock_guard<std::mutex> buf_lk(buffer_mutex_);
            batch.swap(insert_buffer_);
            insert_count_ = 0;
        }

        // 2) replay + clear under a single lock
        {
            std::lock_guard<std::mutex> lk(index_mutex_);
            for (auto &kv : batch) {
                lipp_index_.Insert(kv, /*thread=*/0u);
            }
        }

        flushing_.store(false);
    }

    // underlying indexes
    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType>                               lipp_index_;

    // buffer for pending inserts
    mutable std::mutex                         buffer_mutex_;
    std::vector<KeyValue<KeyType>>             insert_buffer_;
    size_t                                     insert_count_;
    size_t                                     total_insert_count_;
    size_t                                     total_ops_count;
    size_t                                     flush_threshold_;
    std::atomic<bool>                          flushing_;
    std::thread                                flush_thread_;

    // single mutex to guard ALL dp_index_ & lipp_index_ operations
    mutable std::mutex                         index_mutex_;
};