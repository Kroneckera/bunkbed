#!/usr/bin/env julia

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using PercolationOracle
using Printf

const N_RUNS = 3

println("=" ^ 80)
println("Scaling Benchmark: n=3, m=1..8 (2m colorings per condensation)")
println("Julia threads available: $(Threads.nthreads())")
println("Runs per config: $N_RUNS (min time reported)")
println("=" ^ 80)

trees_all = paper_family_decision_trees()
n_obs = 3

# Warm-up
println("\n[warmup] Running paper family regression...")
enumerate_paper_family_regression()

results = Vector{NamedTuple}()

for m in 1:4
    # Use first 2m trees (each tree + complement pair)
    n_colorings = 2 * m
    if n_colorings > length(trees_all)
        println("\n[bench] m=$m requires $(n_colorings) trees, only $(length(trees_all)) available. Stopping.")
        break
    end
    trees_subset = trees_all[1:n_colorings]

    GC.gc()
    times = Float64[]
    total_F = 0
    local all_tuples

    for run in 1:N_RUNS
        all_tuples = Vector{Vector{PartitionID}}()
        t0 = time_ns()

        for g1_partition_id in all_partition_ids(n_obs)
            condensation = condensation_from_partition_id(n_obs, g1_partition_id)
            # First tree (T0) coloring determines the G1-partition, so we skip it
            # and use trees_subset[2:end] as the free colorings
            if n_colorings <= 1
                # m=1: only T0 and T0_complement. T0 gives the condensation partition.
                # Free colorings = [T0_complement]
                colorings = PackedColoring[apply_decision_tree_packed(n_obs, condensation, trees_subset[end])]
            else
                colorings = PackedColoring[apply_decision_tree_packed(n_obs, condensation, tree) for tree in trees_subset[2:end]]
            end
            ctx = PackedOracleContext(n_obs, condensation, colorings)
            suffixes = enumerate_compatible_partitions_packed(ctx)
            for suffix in suffixes
                push!(all_tuples, vcat([g1_partition_id], suffix))
            end
        end

        elapsed = (time_ns() - t0) / 1e9
        push!(times, elapsed)
        total_F = length(all_tuples)
    end

    min_time = minimum(times)
    med_time = sort(times)[cld(length(times), 2)]

    tuple_hash = regression_tuple_hash(n_obs, all_tuples)
    push!(results, (m=m, colorings=n_colorings-1, F_size=total_F, min=min_time, median=med_time, hash=tuple_hash))

    @printf("\n[bench] m=%d (free colorings=%d):\n", m, n_colorings - 1)
    @printf("  |F| = %d\n", total_F)
    @printf("  min = %.6f s  median = %.6f s\n", min_time, med_time)
    @printf("  hash = %s\n", tuple_hash)
end

# Summary table
println("\n" * "=" ^ 80)
println("Scaling Summary (single-threaded)")
println("=" ^ 80)
@printf("%-4s  %-12s  %-12s  %12s  %12s  %-42s\n", "m", "Free cols", "|F|", "Min (s)", "Median (s)", "Hash")
println("-" ^ 80)

for r in results
    @printf("%-4d  %-12d  %-12d  %12.6f  %12.6f  %s\n",
        r.m, r.colorings, r.F_size, r.min, r.median, r.hash)
end

# Also run threaded for the full m=4 case for comparison
if Threads.nthreads() > 1 && length(results) >= 4
    println("\n[bench] m=4 threaded comparison (n_threads=$(Threads.nthreads())):")
    GC.gc()
    times_t = Float64[]
    for run in 1:N_RUNS
        t0 = time_ns()
        enumerate_paper_family_regression_threaded(n_threads=Threads.nthreads())
        push!(times_t, (time_ns() - t0) / 1e9)
    end
    @printf("  Threaded min = %.6f s (vs single-threaded min = %.6f s)\n",
        minimum(times_t), results[4].min)
    @printf("  Speedup = %.2fx\n", results[4].min / minimum(times_t))
end
