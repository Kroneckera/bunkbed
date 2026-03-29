#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using PercolationOracle
using Printf

const N_RUNS = 5

println("=" ^ 70)
println("Threading Benchmark: Paper family (n=3, m=4, 8 colorings)")
println("Julia threads available: $(Threads.nthreads())")
println("Runs per config: $N_RUNS (min time reported)")
println("=" ^ 70)

# Warm-up
println("\n[warmup] Single-threaded...")
enumerate_paper_family_regression()
println("[warmup] Threaded...")
enumerate_paper_family_regression_threaded(n_threads=Threads.nthreads())

results = Dict{Int, NamedTuple}()

for nt in [1, 2, 4]
    if nt > Threads.nthreads()
        println("\n[bench] Skipping n_threads=$nt (only $(Threads.nthreads()) available)")
        continue
    end

    GC.gc()
    times = Float64[]
    local result
    for run in 1:N_RUNS
        t0 = time_ns()
        if nt == 1
            result = enumerate_paper_family_regression()
        else
            result = enumerate_paper_family_regression_threaded(n_threads=nt)
        end
        elapsed = (time_ns() - t0) / 1e9
        push!(times, elapsed)
    end

    min_time = minimum(times)
    med_time = sort(times)[cld(length(times), 2)]
    results[nt] = (count=result.count, hash=result.hash, min=min_time, median=med_time)

    @printf("\n[bench] n_threads=%d:\n", nt)
    @printf("  |F| = %d  hash = %s\n", result.count, result.hash)
    @printf("  min = %.6f s  median = %.6f s\n", min_time, med_time)
    @printf("  counts-by-g1 = %s\n", string(result.counts))
end

# Report speedups
println("\n" * "=" ^ 70)
println("Summary")
println("=" ^ 70)
@printf("%-10s  %12s  %12s  %12s  %12s\n", "Threads", "Min (s)", "Median (s)", "Speedup(min)", "Speedup(med)")
println("-" ^ 70)

base_min = get(results, 1, (min=NaN,)).min
base_med = get(results, 1, (median=NaN,)).median

for nt in sort(collect(keys(results)))
    r = results[nt]
    sp_min = base_min / r.min
    sp_med = base_med / r.median
    @printf("%-10d  %12.6f  %12.6f  %12.2fx %12.2fx\n", nt, r.min, r.median, sp_min, sp_med)
end

# Verify correctness
println("\nCorrectness check:")
ref_hash = results[1].hash
all_match = all(r.hash == ref_hash for r in values(results))
println("  All hashes match reference: $all_match")
println("  Reference hash: $ref_hash")
all_match || error("Hash mismatch detected!")
