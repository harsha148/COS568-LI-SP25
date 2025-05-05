#pragma once

#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>

template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    HybridPGMLIPP(const std::vector<int>& params)
        : dp_index_(params), lipp_index_(params), insert_count_(0), total_insert_count(0),flushing_(false)
    {
        flush_threshold_ = 100000;
    }

    ~HybridPGMLIPP() {
        if (flush_thread_.joinable()) flush_thread_.join();
    }

    uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
        return lipp_index_.Build(data, num_threads);
        total_insert_count = data.size();
    }

    size_t EqualityLookup(const KeyType& key, uint32_t thread_id) const {
        // if (insert_count_ >= flush_threshold_) {
        //     // trigger background flush exactly once
        //     auto self = const_cast<HybridPGMLIPP*>(this);
        //     if (!self->flushing_.exchange(true)) {
        //         if (self->flush_thread_.joinable())
        //             self->flush_thread_.join();
        //         self->flush_thread_ = std::thread(&HybridPGMLIPP::flush_to_lipp, self);
        //     }
        // }
        // if (insert_count_ == 0) {
        //   return lipp_index_.EqualityLookup(key, thread_id);
        // }
        // Fast path for mostly-read workloads
        if (insert_count_ < flush_threshold_/5) { // Only 10% full
            size_t result = lipp_index_.EqualityLookup(key, thread_id);
            if (result != util::NOT_FOUND) return result;
            return dp_index_.EqualityLookup(key, thread_id);
        }
        
        // Normal path
        size_t result = dp_index_.EqualityLookup(key, thread_id);
        return (result == util::OVERFLOW || result == util::NOT_FOUND)
            ? lipp_index_.EqualityLookup(key, thread_id)
            : result;
    }

    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi, uint32_t thread_id) const {
        // if (insert_count_ >= flush_threshold_) {
        //     // trigger background flush exactly once
        //     auto self = const_cast<HybridPGMLIPP*>(this);
        //     if (!self->flushing_.exchange(true)) {
        //         if (self->flush_thread_.joinable())
        //             self->flush_thread_.join();
        //         self->flush_thread_ = std::thread(&HybridPGMLIPP::flush_to_lipp, self);
        //     }
        // }
        return dp_index_.RangeQuery(lo, hi, thread_id) + lipp_index_.RangeQuery(lo, hi, thread_id);
    }

    void Insert(const KeyValue<KeyType>& data, uint32_t thread_id) {
        {
            std::lock_guard<std::mutex> guard(buffer_mutex_);
            insert_buffer_.emplace_back(data);
        }
        dp_index_.Insert(data, thread_id);
        insert_count_++;
        total_insert_count++;
        // Dynamic threshold adjustment
        size_t current_threshold = std::max(flush_threshold_, 
                                          total_insert_count / 20); // Keep DPGM at ~5% of LIPP size

        if (insert_count_ >= current_threshold && !flushing_.exchange(true)) {
            dp_index_ = DynamicPGM<KeyType, SearchClass, pgm_error>(std::vector<int>());
            if (flush_thread_.joinable()) flush_thread_.join();
            flush_thread_ = std::thread(&HybridPGMLIPP::flush_to_lipp, this);
        }
    }

    std::string name() const {
        return "HybridPGMLIPP";
    }

    std::vector<std::string> variants() const {
        return { SearchClass::name(), std::to_string(pgm_error) };
    }

    size_t size() const {
        return dp_index_.size() + lipp_index_.size();
    }

    // Infer insert ratio from ops filename to guide insert behavior
    bool applicable(bool unique, bool range_query, bool insert, bool multithread,
                    const std::string& ops_filename) const {
        return !multithread;
    }

private:
    void flush_to_lipp() {
        std::vector<KeyValue<KeyType>> snapshot;
        {
            std::lock_guard<std::mutex> guard(buffer_mutex_);
            snapshot.swap(insert_buffer_);
            insert_count_ = 0;
        }
        for (const auto& kv : snapshot) {
            lipp_index_.Insert(kv, 0);
        }
        flushing_ = false;
    }

    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType> lipp_index_;

    std::vector<KeyValue<KeyType>> insert_buffer_;
    std::mutex buffer_mutex_;
    size_t insert_count_;
    size_t total_insert_count;
    size_t flush_threshold_;
    mutable std::atomic<bool> flushing_;
    mutable std::thread flush_thread_;

};