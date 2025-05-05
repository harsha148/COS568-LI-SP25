#pragma once

#include "./lipp/src/core/lipp.h"
#include "base.h"

template<class KeyType>
class Lipp: public Base<KeyType>
{
public:
    Lipp(const std::vector<int>& params){}
    uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
        std::vector<std::pair<KeyType, uint64_t>> loading_data;
        loading_data.reserve(data.size());
        for (const auto& itm : data) {
            loading_data.push_back(std::make_pair(itm.key, itm.value));
        }

        return util::timing(
            [&] { lipp_.bulk_load(loading_data.data(), loading_data.size()); });
    }

    size_t EqualityLookup(const KeyType& lookup_key, uint32_t thread_id) const {
        uint64_t value;
        if (!lipp_.find(lookup_key, value)){
            return util::NOT_FOUND;
        }
        return value;
    }

    uint64_t RangeQuery(const KeyType& lower_key, const KeyType& upper_key, uint32_t thread_id) const {
        auto it = lipp_.lower_bound(lower_key);
        uint64_t result = 0;
        while(it != lipp_.end() && it->comp.data.key <= upper_key){
            result += it->comp.data.value;
            ++it;
        }
        return result;
    }

    void Insert(const KeyValue<KeyType>& data, uint32_t thread_id) {
        lipp_.insert(data.key, data.value);
    }

    bool applicable(bool unique, bool range_query, bool insert, bool multithread, const std::string& ops_filename) {
        // LIPP only supports unique keys.
        return unique && !multithread;
    }

    std::string name() const { return "LIPP"; }

    std::vector<std::string> variants() const { 
        return std::vector<std::string>();
    }

    void BulkLoad(const std::vector<KeyValue<KeyType>>& data) {
    if (data.empty()) {
        lipp_.bulk_load(nullptr, 0); // Handle empty case
        return;
    }

    // Create a copy and sort it
    std::vector<std::pair<KeyType, uint64_t>> loading_data;
    loading_data.reserve(data.size());
    for (const auto& itm : data) {
        loading_data.emplace_back(itm.key, itm.value);
    }

    // Sort and remove duplicates
    std::sort(loading_data.begin(), loading_data.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    
    loading_data.erase(std::unique(loading_data.begin(), loading_data.end(),
        [](const auto& a, const auto& b) { return a.first == b.first; }),
        loading_data.end());

    // Verify the sorted data meets LIPP's requirements
    if (!loading_data.empty()) {
        for (size_t i = 1; i < loading_data.size(); ++i) {
            if (loading_data[i].first <= loading_data[i-1].first) {
                std::cerr << "Invalid sort order at position " << i 
                          << ": " << loading_data[i-1].first 
                          << " >= " << loading_data[i].first << "\n";
                throw std::runtime_error("Invalid sort order for LIPP bulk load");
            }
        }
    }

    // Perform the bulk load
    try {
        lipp_.bulk_load(loading_data.data(), loading_data.size());
    } catch (const std::exception& e) {
        std::cerr << "LIPP bulk load failed: " << e.what() << "\n";
        throw;
    }
}

    std::size_t size() const { return lipp_.index_size(); } 
private:
    LIPP<KeyType, uint64_t> lipp_;
};