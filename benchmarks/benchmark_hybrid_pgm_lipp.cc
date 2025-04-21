#include "benchmarks/benchmark_hybrid_pgm_lipp.h"
#include "benchmark.h"
#include "benchmarks/common.h"
#include "competitors/hybrid_pgm_lipp.h"

template <typename Searcher>
void benchmark_64_hybrid_pgm_lipp(tli::Benchmark<uint64_t>& benchmark, 
                                 bool pareto, const std::vector<int>& params) {
  if (!pareto) {
    util::fail("Hybrid PGM-LIPP's hyperparameter cannot be set");
  } else {
    // Test different error bounds for Facebook dataset
    benchmark.template Run<HybridPGMLIPP<uint64_t, Searcher, 16>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, Searcher, 32>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, Searcher, 64>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, Searcher, 128>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, Searcher, 256>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, Searcher, 512>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, Searcher, 1024>>();
  }
}

template <int record>
void benchmark_64_hybrid_pgm_lipp(tli::Benchmark<uint64_t>& benchmark, 
                                 const std::string& filename) {
  // Only run for Facebook dataset
  if (filename.find("fb_100M") != std::string::npos) {
    if (filename.find("0.000000i") != std::string::npos) {
      // Lookup-only workload
      benchmark.template Run<HybridPGMLIPP<uint64_t, LinearSearch<record>, 64>>();
      benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>, 128>>();
      benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>, 64>>();
    } else if (filename.find("mix") == std::string::npos) {
      if (filename.find("0m") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, LinearSearch<record>, 512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>, 512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>, 256>>();
      } else if (filename.find("1m") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>, 1024>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>, 512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>, 256>>();
      } else if (filename.find("2m") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>, 512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>, 512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, LinearSearch<record>, 512>>();
      }
    } else {
      if (filename.find("0.050000i") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, LinearSearch<record>, 64>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>, 128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>, 64>>();
      } else if (filename.find("0.500000i") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>, 128>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>, 64>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>, 512>>();
      } else if (filename.find("0.900000i") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>, 512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>, 64>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>, 128>>();
      } else if (filename.find("0.100000i") != std::string::npos) {
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>, 512>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>, 64>>();
        benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<record>, 128>>();
      }
    }
  }
}

INSTANTIATE_TEMPLATES_MULTITHREAD(benchmark_64_hybrid_pgm_lipp, uint64_t);