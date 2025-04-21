#include "benchmarks/benchmark_hybrid_pgm_lipp.h"
#include "benchmark.h"
#include "benchmarks/common.h"
#include "competitors/hybrid_pgm_lipp.h"

void benchmark_64_hybrid_pgm_lipp(tli::Benchmark<uint64_t>& benchmark) {
  // Only run with default parameters
  benchmark.template Run<HybridPGMLIPP<uint64_t, BranchingBinarySearch<64>, 64>>();
}
