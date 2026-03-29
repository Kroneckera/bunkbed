using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using PercolationOracle
using PercolationOracle: DecisionTree, DecisionTreeStep, PackedColoring, apply_decision_tree_packed, PackedOracleContext, enumerate_compatible_partitions_packed, all_partition_ids, condensation_from_partition_id, PartitionID, regression_tuple_hash, bell_number, normalize_partition_ids, check_partition_compatibility_packed, OracleWorkspace, paper_family_decision_trees
using Base.Threads
using Dates

function extended_decision_trees()
    original_trees = paper_family_decision_trees() # 8 trees
    new_trees = DecisionTree[
        DecisionTree([
            DecisionTreeStep([1], 1),
            DecisionTreeStep([2], 2),
            DecisionTreeStep([3], 1),
            DecisionTreeStep([4], 2),
        ]),
        DecisionTree([
            DecisionTreeStep([1], 2),
            DecisionTreeStep([2], 1),
            DecisionTreeStep([3], 2),
            DecisionTreeStep([4], 1),
        ]),
        DecisionTree([
            DecisionTreeStep([1, 2], 1),
            DecisionTreeStep([3], 2),
            DecisionTreeStep([4], 1),
        ]),
        DecisionTree([
            DecisionTreeStep([1, 2], 2),
            DecisionTreeStep([3], 1),
            DecisionTreeStep([4], 2),
        ]),
        DecisionTree([
            DecisionTreeStep([2, 3], 1),
            DecisionTreeStep([1], 2),
            DecisionTreeStep([4], 1),
        ]),
        DecisionTree([
            DecisionTreeStep([2, 3], 2),
            DecisionTreeStep([1], 1),
            DecisionTreeStep([4], 2),
        ]),
        DecisionTree([
            DecisionTreeStep([1, 3], 1),
            DecisionTreeStep([2], 2),
            DecisionTreeStep([4], 1),
        ]),
        DecisionTree([
            DecisionTreeStep([1, 3], 2),
            DecisionTreeStep([2], 1),
            DecisionTreeStep([4], 2),
        ]),
    ]
    return vcat(original_trees, new_trees)
end

function _backtrack_from_prefix_local!(
    results::Vector{Vector{PartitionID}},
    ctx::PackedOracleContext,
    prefix::Vector{PartitionID},
    workspace::OracleWorkspace,
)
    n_colorings = length(ctx.colorings)
    last_id = PartitionID(bell_number(ctx.n_obs) - 1)
    local_prefix = copy(prefix)
    init_length = length(local_prefix)

    while true
        status = check_partition_compatibility_packed(ctx, local_prefix; workspace=workspace)
        if status && length(local_prefix) < n_colorings
            push!(local_prefix, PartitionID(0))
        else
            if status
                push!(results, copy(local_prefix))
            end
            while length(local_prefix) > init_length && local_prefix[end] == last_id
                pop!(local_prefix)
            end
            if length(local_prefix) == init_length
                break
            end
            local_prefix[end] = PartitionID(Int(local_prefix[end]) + 1)
        end
    end
end

function enumerate_large_family(trees; n_threads::Int=1)
    n_obs = 3
    D = bell_number(n_obs)

    g1_ids = all_partition_ids(n_obs)
    contexts = PackedOracleContext[]
    for g1_pid in g1_ids
        condensation = condensation_from_partition_id(n_obs, g1_pid)
        colorings = PackedColoring[apply_decision_tree_packed(n_obs, condensation, tree) for tree in trees[2:end]]
        push!(contexts, PackedOracleContext(n_obs, condensation, colorings))
    end

    work_items = Tuple{Int, PartitionID}[]
    for (gi, _) in enumerate(g1_ids)
        for sid in PartitionID(0):PartitionID(D - 1)
            push!(work_items, (gi, sid))
        end
    end

    n_items = length(work_items)
    item_results = [Vector{Vector{PartitionID}}() for _ in 1:n_items]
    workspaces = [OracleWorkspace(contexts[1]) for _ in 1:max(1, n_threads)]

    if n_threads == 1
        for item_idx in 1:n_items
            gi, first_suffix = work_items[item_idx]
            _backtrack_from_prefix_local!(item_results[item_idx], contexts[gi], PartitionID[first_suffix], workspaces[1])
        end
    else
        Threads.@threads :dynamic for item_idx in 1:n_items
            gi, first_suffix = work_items[item_idx]
            _backtrack_from_prefix_local!(item_results[item_idx], contexts[gi], PartitionID[first_suffix], workspaces[Threads.threadid()])
        end
    end

    tuples = Vector{Vector{PartitionID}}()
    for item_idx in 1:n_items
        gi = work_items[item_idx][1]
        for suffix in item_results[item_idx]
            push!(tuples, vcat([g1_ids[gi]], suffix))
        end
    end

    tuple_hash = regression_tuple_hash(n_obs, tuples)
    return (
        tuples = tuples,
        hash = tuple_hash,
        count = length(tuples),
    )
end

function main()
    println("Compiling functions...")
    all_trees = extended_decision_trees()
    enumerate_large_family(all_trees[1:4]; n_threads=1)
    enumerate_large_family(all_trees[1:4]; n_threads=Threads.nthreads())
    println("Compilation done.\n")

    configs = [(4, 8), (6, 12), (7, 14)]
    results = []

    for (m, num_trees) in configs
        println("--- Benchmarking m=$(m) ($(num_trees) trees) ---")
        trees_to_use = all_trees[1:num_trees]

        # Single-threaded
        t0 = time()
        res_single = enumerate_large_family(trees_to_use; n_threads=1)
        t1 = time()
        time_single = round(t1 - t0, digits=4)

        # Multi-threaded
        t0 = time()
        res_threaded = enumerate_large_family(trees_to_use; n_threads=Threads.nthreads())
        t1 = time()
        time_threaded = round(t1 - t0, digits=4)

        println("  |F| count: \t\t", res_single.count)
        println("  Single-thread time: \t", time_single, " s")
        println("  Multi-thread time: \t", time_threaded, " s")
        if time_threaded > 0
            println("  Speedup: \t\t", round(time_single / time_threaded, digits=2), "x")
        end
        println("  Hash (single): \t", res_single.hash)
        
        if res_single.count == res_threaded.count && res_single.hash == res_threaded.hash
            println("  Consistency check: \tPASS")
        else
            println("  Consistency check: \tFAIL")
            exit(1)
        end
        
        # We save tuples as Set for subset verification
        push!(results, (m=m, tuples=Set(res_single.tuples)))
        println()
    end

    println("--- Correctness Verification (Subset Consistency) ---")
    for i in 1:(length(results)-1)
        m_curr = results[i].m
        set_curr = results[i].tuples
        
        m_next = results[i+1].m
        set_next = results[i+1].tuples

        # The result for m_next (more trees) should be compatible with m_curr (fewer trees)
        # Specifically, if a tuple is valid for m_next, its prefix corresponding to m_curr must be valid for m_curr.
        # Let's project the tuples in set_next to their prefixes of length for m_curr.
        num_colorings_curr = length(all_trees[2:(configs[i][2])])
        # prefix length is 1 (g1) + num_colorings_curr
        len_curr = 1 + num_colorings_curr

        projected = Set(t[1:len_curr] for t in set_next)
        
        is_sub = issubset(projected, set_curr)
        println("Is projected F(m=$(m_next)) a subset of F(m=$(m_curr))? \t", is_sub ? "PASS" : "FAIL")
        if !is_sub
            println("Subset verification failed!")
            exit(1)
        end
    end
    println("\nAll benchmarks and verifications completed successfully.")
end

main()
