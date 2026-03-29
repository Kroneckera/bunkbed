using SHA

export PAPER_FAMILY_LABELS, paper_family_decision_trees
export regression_tuple_line, regression_tuple_hash, enumerate_paper_family_regression

const PAPER_FAMILY_LABELS = (
    "T0",
    "T0_complement",
    "T2",
    "T2_complement",
    "T1",
    "T1_complement",
    "T3",
    "T3_complement",
)

function paper_family_decision_trees()
    return DecisionTree[
        DecisionTree([DecisionTreeStep([1, 2, 3, 4], 1)]),
        DecisionTree([DecisionTreeStep([1, 2, 3, 4], 2)]),
        DecisionTree([
            DecisionTreeStep([2], 1),
            DecisionTreeStep([1], 2),
            DecisionTreeStep([3], 1),
            DecisionTreeStep([4], 2),
        ]),
        DecisionTree([
            DecisionTreeStep([2], 2),
            DecisionTreeStep([1], 1),
            DecisionTreeStep([3], 2),
            DecisionTreeStep([4], 1),
        ]),
        DecisionTree([
            DecisionTreeStep([3], 1),
            DecisionTreeStep([1], 2),
            DecisionTreeStep([2], 1),
            DecisionTreeStep([4], 2),
        ]),
        DecisionTree([
            DecisionTreeStep([3], 2),
            DecisionTreeStep([1], 1),
            DecisionTreeStep([2], 2),
            DecisionTreeStep([4], 1),
        ]),
        DecisionTree([
            DecisionTreeStep([1], 1),
            DecisionTreeStep([2, 3], 2),
            DecisionTreeStep([4], 1),
        ]),
        DecisionTree([
            DecisionTreeStep([1], 2),
            DecisionTreeStep([2, 3], 1),
            DecisionTreeStep([4], 2),
        ]),
    ]
end

function regression_tuple_line(n_obs::Int, tuple_ids::AbstractVector{<:Integer})
    return join((partition_label(n_obs, pid) for pid in tuple_ids), " || ")
end

function regression_tuple_hash(n_obs::Int, tuples::AbstractVector{<:AbstractVector{<:Integer}})
    lines = sort!(String[join(string.(Int.(tuple) .+ 1), "\t") for tuple in tuples])
    return bytes2hex(sha1(join(lines, "\n")))
end

function enumerate_paper_family_regression()
    n_obs = 3
    trees = paper_family_decision_trees()
    tuples = Vector{Vector{PartitionID}}()
    counts = Int[]

    for g1_partition_id in all_partition_ids(n_obs)
        condensation = condensation_from_partition_id(n_obs, g1_partition_id)
        colorings = PackedColoring[apply_decision_tree_packed(n_obs, condensation, tree) for tree in trees[2:end]]
        ctx = PackedOracleContext(n_obs, condensation, colorings)
        suffixes = enumerate_compatible_partitions_packed(ctx)
        push!(counts, length(suffixes))
        for suffix in suffixes
            push!(tuples, vcat([g1_partition_id], suffix))
        end
    end

    tuple_hash = regression_tuple_hash(n_obs, tuples)
    return (
        tuples = tuples,
        counts = counts,
        hash = tuple_hash,
        count = length(tuples),
    )
end
