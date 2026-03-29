using Test
using Random
using Serialization
using PercolationOracle

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PKG_ROOT = normpath(joinpath(@__DIR__, ".."))
if !isdefined(Main, :DecisionTreeAlgorithm)
    include(joinpath(REPO_ROOT, "legacy", "decision_trees.jl"))
end
using .DecisionTreeAlgorithm

function baseline_tree(tree::PercolationOracle.DecisionTree)
    return DecisionTreeAlgorithm.DecisionTree{Int}([
        DecisionTreeAlgorithm.DecisionTreeStep(copy(step.vertices), Int(step.graph_id)) for step in tree
    ])
end

function baseline_colorings(n_obs::Int, condensation::PercolationOracle.CondensationBits, trees::AbstractVector{<:PercolationOracle.DecisionTree})
    matrix = decode_condensation(n_obs, condensation)
    return [DecisionTreeAlgorithm.apply_decision_tree(matrix, baseline_tree(tree)) for tree in trees]
end

function baseline_check(condensation::PercolationOracle.CondensationBits, colorings::AbstractVector{PercolationOracle.PackedColoring}, partition_ids::AbstractVector{<:Integer}; n_obs::Int=3)
    partitions = [decode_partition(n_obs, pid) for pid in partition_ids]
    matrices = [unpack_coloring(n_obs, color) for color in colorings]
    return DecisionTreeAlgorithm.check_partition_compatibility(decode_condensation(n_obs, condensation), matrices, partitions)
end

function baseline_universal_graph(condensation::PercolationOracle.CondensationBits, colorings::AbstractVector{PercolationOracle.PackedColoring}, partition_ids::AbstractVector{<:Integer}; n_obs::Int=3)
    partitions = [decode_partition(n_obs, pid) for pid in partition_ids]
    matrices = [unpack_coloring(n_obs, color) for color in colorings]
    return DecisionTreeAlgorithm.UniversalGraph(decode_condensation(n_obs, condensation), matrices, partitions), matrices
end

function packed_neighbors(ctx::PackedOracleContext, partition_ids::AbstractVector{<:Integer}, state::PercolationOracle.PackedState, active_index::Int)
    normalized = PercolationOracle.normalize_partition_ids(partition_ids)
    m = length(normalized)
    neighbors = UInt64[]
    component = component_from_state(state)
    for next_component in 1:ctx.n_total
        next_state, ok = if component == ctx.n_total && next_component == ctx.n_total
            PercolationOracle.build_star_loop_state(ctx, UInt64(state), active_index, m), true
        else
            PercolationOracle.build_nonstar_state(ctx, normalized, UInt64(state), component, next_component, active_index, m)
        end
        ok && push!(neighbors, UInt64(next_state))
    end
    sort!(unique!(neighbors))
    return neighbors
end

function baseline_neighbors(ug, matrices, component::Int, psi_values::AbstractVector{<:Integer}, active_index::Int)
    code = DecisionTreeAlgorithm.CodeMask((component, collect(Int.(psi_values))))
    neighbors = DecisionTreeAlgorithm.get_masking_neighbors(ug, code, matrices[active_index])
    packed = UInt64[]
    for neighbor in neighbors
        push!(packed, UInt64(pack_state(neighbor.component, neighbor.ids)))
    end
    sort!(unique!(packed))
    return packed
end

function one_based_tuples(zero_based_tuples)
    return [Int.(tuple) .+ 1 for tuple in zero_based_tuples]
end
