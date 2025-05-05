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
#include <algorithm>
#include <iostream>

template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    HybridPGMLIPP(const std::vector<int>& params)
        : dp_index_(params), lipp_index_(params), insert_count_(0), flushing_(false) {
        flush_threshold_ = 10000; // Default threshold
    }

    ~HybridPGMLIPP() {
        // Ensure flush completes before destruction
        if (flushing_) {
            if (flush_thread_.joinable()) {
                flush_thread_.join();
            }
        } else if (flush_thread_.joinable()) {
            flush_thread_.join();
        }
        
        // Clear buffers
        {
            std::lock_guard<std::mutex> guard(buffer_mutex_);
            insert_buffer_.clear();
        }
    }

    uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
        // Clear any existing state
        {
            std::lock_guard<std::mutex> guard(buffer_mutex_);
            insert_buffer_.clear();
        }
        insert_count_ = 0;
        flushing_ = false;
        
        // Reset both indexes
        dp_index_ = DynamicPGM<KeyType, SearchClass, pgm_error>(std::vector<int>());
        
        // Directly bulk load to LIPP for initial build
        uint64_t build_time = util::timing([&] {
            lipp_index_.BulkLoad(data);
        });
        
        return build_time;
    }

    size_t EqualityLookup(const KeyType& key, uint32_t thread_id) const {
        // Fast path for mostly-read workloads
        if (insert_count_ < flush_threshold_/10) {
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
        // Prefetch both index ranges
        __builtin_prefetch(&dp_index_, 0, 0);
        __builtin_prefetch(&lipp_index_, 0, 0);
        
        return dp_index_.RangeQuery(lo, hi, thread_id) + 
               lipp_index_.RangeQuery(lo, hi, thread_id);
    }

    void Insert(const KeyValue<KeyType>& data, uint32_t thread_id) {
        {
            std::lock_guard<std::mutex> guard(buffer_mutex_);
            // Find insertion position to maintain sort order
            auto it = std::lower_bound(insert_buffer_.begin(), insert_buffer_.end(), data,
                [](const KeyValue<KeyType>& a, const KeyValue<KeyType>& b) {
                    return a.key < b.key;
                });
            insert_buffer_.insert(it, data);
        }
        
        dp_index_.Insert(data, thread_id);
        insert_count_++;
        stats_.total_inserts++;

        // Dynamic threshold adjustment based on LIPP size
        size_t current_threshold = std::max<size_t>(
            flush_threshold_, 
            lipp_index_.size() ? lipp_index_.size() / 10 : 10000
        );

        if (insert_count_ >= current_threshold && !flushing_.exchange(true)) {
            dp_index_ = DynamicPGM<KeyType, SearchClass, pgm_error>(std::vector<int>());
            if (flush_thread_.joinable()) {
                flush_thread_.join();
            }
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

    bool applicable(bool unique, bool range_query, bool insert, bool multithread,
                    const std::string& ops_filename) const {
        return !multithread;
    }

    void PrintStats() const {
        std::cout << "Hybrid Index Stats:\n"
                  << "  Total inserts: " << stats_.total_inserts << "\n"
                  << "  Successful flushes: " << stats_.successful_flushes << "\n"
                  << "  Failed flushes: " << stats_.failed_flushes << "\n"
                  << "  Items flushed: " << stats_.items_flushed << "\n"
                  << "  Current buffer size: " << insert_buffer_.size() << "\n"
                  << "  DPGM size: " << dp_index_.size() << "\n"
                  << "  LIPP size: " << lipp_index_.size() << std::endl;
    }

private:
    void flush_to_lipp() {
        std::vector<KeyValue<KeyType>> snapshot;
        {
            std::lock_guard<std::mutex> guard(buffer_mutex_);
            if (insert_buffer_.empty()) {
                flushing_ = false;
                return;
            }
            snapshot.swap(insert_buffer_);
            insert_count_ = 0;
        }

        // Sort and prepare data for LIPP
        std::sort(snapshot.begin(), snapshot.end(),
            [](const auto& a, const auto& b) { return a.key < b.key; });

        // Remove duplicates if needed
        snapshot.erase(std::unique(snapshot.begin(), snapshot.end(),
            [](const auto& a, const auto& b) { return a.key == b.key; }),
            snapshot.end());

        // Safe bulk load with error handling
        try {
            lipp_index_.BulkLoad(snapshot);
            stats_.successful_flushes++;
            stats_.items_flushed += snapshot.size();
        } catch (const std::exception& e) {
            std::cerr << "Failed to bulk load to LIPP: " << e.what() << std::endl;
            stats_.failed_flushes++;
            // Re-insert failed items back to buffer
            std::lock_guard<std::mutex> guard(buffer_mutex_);
            insert_buffer_.insert(insert_buffer_.end(), 
                                std::make_move_iterator(snapshot.begin()),
                                std::make_move_iterator(snapshot.end()));
            insert_count_ = snapshot.size();
        }
        
        flushing_ = false;
    }

    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType> lipp_index_;

    std::vector<KeyValue<KeyType>> insert_buffer_;
    std::mutex buffer_mutex_;
    size_t insert_count_;
    size_t flush_threshold_;
    mutable std::atomic<bool> flushing_;
    mutable std::thread flush_thread_;

    struct {
        size_t total_inserts = 0;
        size_t successful_flushes = 0;
        size_t failed_flushes = 0;
        size_t items_flushed = 0;
    } stats_;
};