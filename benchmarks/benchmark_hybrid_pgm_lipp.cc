#include "benchmarks/benchmark_hybrid_pgm_lipp.h"
#include "benchmark.h"
#include "benchmarks/common.h"
#include "competitors/hybrid_pgm_lipp.h"

// 1) Pareto/hyperparameter sweep overload
template <typename Searcher>
void benchmark_64_hybrid_pgm_lipp(tli::Benchmark<uint64_t>& benchmark,
                                  bool pareto,
                                  const std::vector<int>& params) {
  if (!pareto) {
    util::fail("HybridPGMLipp's hyperparameter cannot be set");
  } else {
    // We ignore params for Milestone 2 and fix pgm_error=16
    benchmark.template Run<HybridPGMLIPP<uint64_t, Searcher, 16>>(params);
  }
}

// 2) Filename‐based overload (only run on FB dataset here)
template <int record>
void benchmark_64_hybrid_pgm_lipp(tli::Benchmark<uint64_t>& benchmark,
                                  const std::string& filename) {
  if (filename.find("fb_100M") != std::string::npos) {
    benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>, 256>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>, 512>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, ExponentialSearch<record>, 128>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>, 256>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>, 512>>();
    benchmark.template Run<HybridPGMLIPP<uint64_t, InterpolationSearch<record>, 128>>();

  }
}

// Instantiate for track_errors = 0,1,2
INSTANTIATE_TEMPLATES_MULTITHREAD(benchmark_64_hybrid_pgm_lipp, uint64_t);