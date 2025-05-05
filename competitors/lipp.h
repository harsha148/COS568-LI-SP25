#pragma once

#include "./lipp/src/core/lipp.h"
#include "base.h"

#include <thread>
#include <vector>
#include <algorithm>

template<class KeyType>
class Lipp: public Base<KeyType>
{
public:
    Lipp(const std::vector<int>& params) {}

    uint64_t Build(const std::vector<KeyValue<KeyType>>& data,
                   size_t num_threads)
    {
        std::vector<std::pair<KeyType, uint64_t>> loading;
        loading.reserve(data.size());
        for (auto &itm : data) {
            loading.emplace_back(itm.key, itm.value);
        }
        return lipp_.bulk_load(loading.data(), int(loading.size()));
    }

    void Insert(const KeyValue<KeyType>& d,
                uint32_t thread_id)
    {
        lipp_.insert(d.key, d.value);
    }

    // NEW – a parallelized bulk‐insert that shards the work across cores
    void BulkInsert(const std::vector<KeyValue<KeyType>>& data,
                    uint32_t /*thread_id*/)
    {
        const size_t N = data.size();
        size_t nThreads = std::thread::hardware_concurrency();
        if (nThreads == 0) nThreads = 1;
        size_t chunk = (N + nThreads - 1) / nThreads;

        std::vector<std::thread> workers;
        workers.reserve(nThreads);
        for (size_t t = 0; t < nThreads; ++t) {
            size_t start = t * chunk;
            size_t end   = std::min(start + chunk, N);
            if (start >= end) break;

            workers.emplace_back([&, start, end]() {
                for (size_t i = start; i < end; ++i) {
                    const auto &kv = data[i];
                    lipp_.insert(kv.key, kv.value);
                }
            });
        }
        for (auto &th : workers) th.join();
    }

    size_t EqualityLookup(const KeyType& key,
                          uint32_t thread_id) const
    {
        return lipp_.at(key);
    }

    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi,
                        uint32_t thread_id) const
    {
        // No native range‐query, so fall back to scanning around lo/hi
        // (or implement an iterator-based scan if you prefer)
        uint64_t count = 0;
        for (auto it = lipp_.lower_bound(lo); it != lipp_.end(); ++it) {
            if (it->comp.data.key > hi) break;
            ++count;
        }
        return count;
    }

    std::string name() const { return "LIPP"; }
    std::vector<std::string> variants() const { return {}; }
    size_t size() const { return lipp_.index_size(); }

private:
    LIPP<KeyType, uint64_t> lipp_;
};