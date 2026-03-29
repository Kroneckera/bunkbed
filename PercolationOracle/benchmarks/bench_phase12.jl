#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using PercolationOracle
using Printf

const BASELINE_WALL_SECONDS = 29.1005

println("[bench_phase12] Warm-up run...")
warmup = enumerate_paper_family_regression()
println("[bench_phase12] Warm-up count=$(warmup.count) hash=$(warmup.hash)")

GC.gc()
start_ns = time_ns()
result = enumerate_paper_family_regression()
elapsed = (time_ns() - start_ns) / 1e9
speedup = BASELINE_WALL_SECONDS / elapsed

println("[bench_phase12] Regression count=$(result.count)")
println("[bench_phase12] Regression counts-by-g1=$(result.counts)")
println("[bench_phase12] Regression hash=$(result.hash)")
@printf("[bench_phase12] Optimized wall time: %.6f s\n", elapsed)
@printf("[bench_phase12] Baseline wall time reference: %.4f s\n", BASELINE_WALL_SECONDS)
@printf("[bench_phase12] Speedup vs baseline reference: %.2fx\n", speedup)
