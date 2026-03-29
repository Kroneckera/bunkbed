module DecisionTreeAlgorithm

using Combinatorics
using DataStructures
using ProgressMeter

export Partition, DecisionTreeStep, DecisionTree
export apply_decision_tree, check_partition_compatibility
export partition_repr, enumerate_compatible_partitions


Partition{T} = Vector{Vector{T}}

DecisionTreeStep{T} = @NamedTuple{vertices::Vector{T}, graph_id::Int}
function DecisionTreeStep(vertices::Vector{T}, graph_id::Int) where T
    return DecisionTreeStep{T}((vertices, graph_id=graph_id))
end

DecisionTree{T} = Vector{DecisionTreeStep{T}}

"""
    partition_repr(partition::Partition{T}) where T

Returns the string representation of the partition.

# Examples
```julia-repl
julia> partition_repr([[1, 2], [4], [3]])
"12|4|3"
```
"""
function partition_repr(partition::Partition{T}) where T
    return join([join([item for item ∈ block]) for block ∈ partition], "|")
end

"""
    apply_decision_tree(graph::Matrix{Bool}, decision_tree::DecisionTree{Int})

For a given condensation graph `g1_condensation` and a decision tree `decision_tree`,
returns the partition (or coloring) of the components of `g1_condensation`
and edges between them according to the decision tree.

# Arguments
- `g1_condensation::Matrix{Bool}`: the adjacency matrix of the condensation graph of `g1`.
- `decision_tree::DecisionTree{Int}`: the decision tree to be applied.

# Returns
- `coloring::Matrix{Int}`: the partition of components of `g1_condensation` 
    and edges between them according to the decision tree.
"""
function apply_decision_tree(g1_condensation::Matrix{Bool}, decision_tree::DecisionTree{Int})
    coloring = zeros(Int, size(g1_condensation))
    for step in decision_tree
        queue = Queue{Int}()
        for u in step.vertices
            enqueue!(queue, u)
        end
        
        while !isempty(queue)
            u = dequeue!(queue)
            if coloring[u, u] > 0
                continue
            end
            coloring[u, u] = step.graph_id
            
            for v in eachindex(g1_condensation[u, :])
                if u == v || coloring[u, v] > 0
                    continue
                end
                coloring[u, v] = step.graph_id
                coloring[v, u] = step.graph_id

                if g1_condensation[u, v]
                    enqueue!(queue, v)
                end
            end
        end
    end
    return coloring
end

struct UniversalGraph
    g1_condensation::Matrix{Bool}
    ranges::Vector{Int}
    masks::Array{Int, 2}
    colorings::Array{Int, 3}
    selected_vertices::Array{Int, 2}
end


function UniversalGraph(
        g1_condensation::Matrix{Bool},
        colorings::Vector{Matrix{Int}},
        partitions::Vector{Partition{Int}})
    k = length(colorings)
    n = size(colorings[1], 1)

    selected_vertices = zeros(Int, n, k)
    ranges = zeros(Int, k)
    masks = zeros(Int, n, k)

    for (id, (coloring, partition)) in enumerate(zip(colorings, partitions))
        ranges[id] = length(partition)
        for (i, block) in enumerate(partition)
            for u in block
                selected_vertices[u, id] = i
                masks[u, id] = coloring[u, u] == 1 ? i : 0 
            end
        end
    end

    return UniversalGraph(
        g1_condensation, ranges, masks,
        cat(colorings..., dims=3), selected_vertices
    )
end

"""
    CodeMask = @NamedTuple{component::Int, ids::Vector{Int}}

A named tuple to represent the code mask of a vertex in the universal graph.
The `component` field is the index of the component in the partition of the condensation graph.
The `ids` field repesents the mask of the vertex in the universal graph.
If `ids[i] = j > 0`, then the vertex belongs to the `j`-th partition block of the `i`-th decision tree.
The mask `ids[i] = 0`, shows that corresponding vertex can be from any partition block of the `i`-th decision tree.
"""
CodeMask = @NamedTuple{component::Int, ids::Vector{Int}}

"""
    get_masking_neighbors(universal_graph::UniversalGraph, code::CodeMask, coloring::Matrix{Int})

Returns the neighbors of the given `code` mask in the `universal_graph`
after the decision tree is applied.

# Arguments
- `universal_graph::UniversalGraph`: the universal graph.
- `code::CodeMask`: the code mask of the vertex.
- `coloring::Matrix{Int}`: the partition of the condensation graph
    between the G1 and G2 according to the decision tree.

# Returns
- `neighbors::Vector{CodeMask}`: the neighbors of the given `code` mask in the `universal_graph`.
"""
function get_masking_neighbors(universal_graph::UniversalGraph, code::CodeMask, coloring::Matrix{Int})
    k = length(code.ids)
    n = size(universal_graph.colorings, 1)

    u = code.component
    neighbors = Vector{CodeMask}()
    for v in 1:n
        neighbor = CodeMask((v, zeros(Int, k)))

        # First, we check the case when the vertices are from the component of remaining vertices
        if u == n && v == n
            neighbor.ids .= ifelse.(
                universal_graph.colorings[n, n, :] .== coloring[u, v],
                code.ids, zeros(Int, k)
            )
        elseif coloring[u, v] == 1
            if universal_graph.g1_condensation[u, v] || (u == v && u ≠ n)
                neighbor.ids .= universal_graph.masks[v, :]
            else
                continue
            end
        elseif coloring[u, v] == 2
            if any((universal_graph.colorings[u, v, :] .== 2)
                    .& (code.ids .≠ 0) .& (universal_graph.masks[v, :] .≠ 0)
                    .& (universal_graph.masks[v, :] .!= code.ids))
                continue
            end
            neighbor.ids .= ifelse.(
                universal_graph.colorings[u, v, :] .== 1,
                universal_graph.masks[v, :],
                max.(universal_graph.masks[v, :], code.ids)
            )
        end
        
        push!(neighbors, neighbor)
    end
    return neighbors
end

function check_g1_compatibility(
        g1_condensation::Matrix{Bool},
        coloring::Matrix{Int},
        partition::Partition{Int},
    )
    for (block1, block2) in combinations(partition, 2), u in block1, v in block2
        if coloring[u, v] == 1 && g1_condensation[u, v]
            return false
        end
    end
    return true
end


"""
    check_universal_graph_partition(universal_graph::UniversalGraph, coloring, partition)

Checks if the given `partition` of the decision tree represented by the `coloring`
is compatible with the constructed universal graph.

# Arguments
- `universal_graph::UniversalGraph`: the universal graph.
- `coloring::Matrix{Int}`: the partition of the condensation graph
between the G1 and G2 according to the decision tree.
- `partition::Partition{Int}`: the partition of the selected vertices between components

"""
function check_universal_graph_partition(universal_graph::UniversalGraph, coloring, partition)
    for block in partition
        u = block[1]
        code = CodeMask((u, universal_graph.selected_vertices[u, :]))

        visited = [false for _ in eachindex(block)]
        visited[1] = true
        
        used = Set{CodeMask}([code])
        queue = Deque{CodeMask}()
        push!(queue, code)
        
        while !isempty(queue)
            code = popfirst!(queue)
            for next_code in get_masking_neighbors(universal_graph, code, coloring)
                if next_code in used
                    continue
                end
                push!(used, next_code)
                push!(queue, next_code)
                for (id, v) in enumerate(block)
                    if visited[id] || next_code.component ≠ v
                        continue
                    end
                    target = universal_graph.selected_vertices[v, :]
                    visited[id] = all((next_code.ids .== 0) .| (target .== next_code.ids))
                end
                if all(visited)
                    break
                end
            end
        end
        if !all(visited)
            return false
        end
    end
    return true
end

function check_partition_compatibility(
        g1_condensation::Matrix{Bool},
        colorings::Vector{Matrix{Int}},
        partitions::Vector{Partition{Int}},
    )
    if !all(check_g1_compatibility.(Ref(g1_condensation), colorings, partitions))
        return false
    end

    universal_graph = UniversalGraph(g1_condensation, colorings, partitions)
    return all(check_universal_graph_partition.(Ref(universal_graph), colorings, partitions))
end

function enumerate_compatible_partitions(
            g1_condensation::Matrix{Bool},
            colorings::Vector{Matrix{Int}},
            prefix::Vector{T}
        ) where T <: Integer
    n = size(g1_condensation, 1) - 1
    set_partitions = collect(partitions(1:n))
    current_prefix = copy(prefix)
    compatible_partitions = Vector{T}[]
    init_length = length(current_prefix)

    D = length(set_partitions)
    progress = Progress(D ^ (length(colorings) - init_length))

    while true
        status = isempty(current_prefix) || check_partition_compatibility(
            g1_condensation,
            colorings[1:length(current_prefix)],
            [set_partitions[id] for id ∈ current_prefix],
        )
        if status && length(current_prefix) < length(colorings)
            push!(current_prefix, 1)
        else
            if status
                push!(compatible_partitions, copy(current_prefix))
            end
            while length(current_prefix) > init_length && current_prefix[end] == length(set_partitions)
                pop!(current_prefix)
            end
            if length(current_prefix) == init_length
                break
            end
            current_prefix[end] += 1
            ProgressMeter.update!(progress, sum(
                current_prefix[id] * D ^ (length(colorings) - init_length - id)
                for id in eachindex(current_prefix)
            ))
        end
    end
    return compatible_partitions
end

end