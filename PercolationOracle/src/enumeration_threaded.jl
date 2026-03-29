export enumerate_compatible_partitions_threaded
export enumerate_paper_family_regression_threaded

"""
    _backtrack_from_prefix!(results, ctx, prefix, workspace)

Run backtracking enumeration from `prefix`, appending all compatible full-length
partition tuples to `results`. Uses the provided `workspace` for BFS state.
"""
function _backtrack_from_prefix!(
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

"""
    enumerate_compatible_partitions_threaded(ctx, n_threads; prefix=[])

Multi-threaded version of `enumerate_compatible_partitions_packed`.
Partitions work at the first free partition level: each work item is a
first-partition ID, processed independently with its own per-item result
vector. Thread-local workspaces avoid allocation contention.
"""
function enumerate_compatible_partitions_threaded(
    ctx::PackedOracleContext,
    n_threads::Int;
    prefix::AbstractVector{<:Integer}=PartitionID[],
)
    n_threads >= 1 || throw(ArgumentError("n_threads must be >= 1"))
    D = bell_number(ctx.n_obs)
    n_colorings = length(ctx.colorings)
    current_prefix = normalize_partition_ids(prefix)
    init_length = length(current_prefix)

    # If the prefix already covers all colorings, just check it
    if init_length >= n_colorings
        ws = OracleWorkspace(ctx)
        if check_partition_compatibility_packed(ctx, current_prefix[1:n_colorings]; workspace=ws)
            return [current_prefix[1:n_colorings]]
        else
            return Vector{Vector{PartitionID}}()
        end
    end

    # If only one level left or single thread requested, fall back to single-threaded
    if n_threads == 1 || init_length >= n_colorings - 1
        return enumerate_compatible_partitions_packed(ctx, current_prefix)
    end

    # Pre-filter: if prefix is non-empty, check that prefix is still compatible
    if !isempty(current_prefix)
        ws = OracleWorkspace(ctx)
        if !check_partition_compatibility_packed(ctx, current_prefix; workspace=ws)
            return Vector{Vector{PartitionID}}()
        end
    end

    n_items = D

    # Per-item result storage (lock-free: each item writes only to its own vector)
    item_results = [Vector{Vector{PartitionID}}() for _ in 1:n_items]

    # One workspace per OS thread (critical: :dynamic can schedule on any thread)
    workspaces = [OracleWorkspace(ctx) for _ in 1:Threads.nthreads()]

    Threads.@threads :dynamic for item_idx in 1:n_items
        local_ws = workspaces[Threads.threadid()]
        extended = vcat(current_prefix, [PartitionID(item_idx - 1)])
        _backtrack_from_prefix!(item_results[item_idx], ctx, extended, local_ws)
    end

    # Merge results in deterministic order
    result = Vector{Vector{PartitionID}}()
    sizehint!(result, sum(length(r) for r in item_results))
    for r in item_results
        append!(result, r)
    end
    return result
end

"""
    enumerate_paper_family_regression_threaded(; n_threads=Threads.nthreads())

Multi-threaded version of `enumerate_paper_family_regression`.
Generates all (g1_partition_id, first_suffix_id) work items and distributes
them across threads for maximum parallelism and load balance.
"""
function enumerate_paper_family_regression_threaded(; n_threads::Int=Threads.nthreads())
    n_obs = 3
    trees = paper_family_decision_trees()
    D = bell_number(n_obs)

    # Pre-compute oracle contexts for each G1-partition
    g1_ids = all_partition_ids(n_obs)
    contexts = PackedOracleContext[]
    for g1_pid in g1_ids
        condensation = condensation_from_partition_id(n_obs, g1_pid)
        colorings = PackedColoring[apply_decision_tree_packed(n_obs, condensation, tree) for tree in trees[2:end]]
        push!(contexts, PackedOracleContext(n_obs, condensation, colorings))
    end

    # Generate all work items: (g1_index, first_suffix_id)
    work_items = Tuple{Int, PartitionID}[]
    for (gi, _) in enumerate(g1_ids)
        for sid in PartitionID(0):PartitionID(D - 1)
            push!(work_items, (gi, sid))
        end
    end

    n_items = length(work_items)

    # Per-item storage (lock-free: deterministic order)
    item_results = [Vector{Vector{PartitionID}}() for _ in 1:n_items]

    # One workspace per OS thread
    workspaces = [OracleWorkspace(contexts[1]) for _ in 1:Threads.nthreads()]

    if n_threads == 1
        for item_idx in 1:n_items
            gi, first_suffix = work_items[item_idx]
            _backtrack_from_prefix!(item_results[item_idx], contexts[gi], PartitionID[first_suffix], workspaces[1])
        end
    else
        Threads.@threads :dynamic for item_idx in 1:n_items
            gi, first_suffix = work_items[item_idx]
            _backtrack_from_prefix!(item_results[item_idx], contexts[gi], PartitionID[first_suffix], workspaces[Threads.threadid()])
        end
    end

    # Merge in deterministic order
    counts = zeros(Int, length(g1_ids))
    tuples = Vector{Vector{PartitionID}}()
    for item_idx in 1:n_items
        gi = work_items[item_idx][1]
        for suffix in item_results[item_idx]
            counts[gi] += 1
            push!(tuples, vcat([g1_ids[gi]], suffix))
        end
    end

    tuple_hash = regression_tuple_hash(n_obs, tuples)
    return (
        tuples = tuples,
        counts = collect(counts),
        hash = tuple_hash,
        count = length(tuples),
    )
end
