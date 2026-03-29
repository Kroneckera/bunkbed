export PackedColoring, DecisionTreeStep, DecisionTree
export pack_coloring, unpack_coloring, coloring_value, packed_coloring_index
export apply_decision_tree_matrix, apply_decision_tree_packed

const PackedColoring = UInt16

struct DecisionTreeStep
    vertices::Vector{Int}
    graph_id::UInt8
    function DecisionTreeStep(vertices::AbstractVector{<:Integer}, graph_id::Integer)
        g = UInt8(graph_id)
        g in (UInt8(1), UInt8(2)) || throw(ArgumentError("graph_id must be 1 or 2"))
        normalized = sort!(unique(Int.(collect(vertices))))
        isempty(normalized) && throw(ArgumentError("decision-tree steps must have at least one seed vertex"))
        return new(normalized, g)
    end
end

const DecisionTree = Vector{DecisionTreeStep}

function build_coloring_index_table(n_total::Int)
    table = zeros(UInt8, n_total, n_total)
    idx = UInt8(1)
    for u in 1:n_total
        for v in u:n_total
            table[u, v] = idx
            table[v, u] = idx
            idx += UInt8(1)
        end
    end
    return table
end

const COLORING_INDEX_TABLE = Dict{Int,Matrix{UInt8}}(n_total => build_coloring_index_table(n_total) for n_total in 2:(MAX_OBSERVABLES + 1))

packed_coloring_index(n_obs::Int, u::Int, v::Int) = Int(COLORING_INDEX_TABLE[n_obs + 1][u, v])

function pack_coloring(matrix::AbstractMatrix{<:Integer})::PackedColoring
    n_total = size(matrix, 1)
    n_obs = n_total - 1
    bits = zero(PackedColoring)
    for u in 1:n_total
        for v in u:n_total
            if Int(matrix[u, v]) == 2
                idx = packed_coloring_index(n_obs, u, v)
                bits |= PackedColoring(1) << (idx - 1)
            end
        end
    end
    return bits
end

function unpack_coloring(n_obs::Int, coloring::PackedColoring)
    n_total = n_obs + 1
    matrix = Matrix{Int}(undef, n_total, n_total)
    for u in 1:n_total, v in 1:n_total
        matrix[u, v] = Int(coloring_value(coloring, n_obs, u, v))
    end
    return matrix
end

function coloring_value(coloring::PackedColoring, n_obs::Int, u::Int, v::Int)::UInt8
    idx = packed_coloring_index(n_obs, u, v)
    return ((coloring >> (idx - 1)) & PackedColoring(1)) == PackedColoring(1) ? UInt8(2) : UInt8(1)
end

function apply_decision_tree_matrix(n_obs::Int, condensation::CondensationBits, tree::DecisionTree)
    n_total = n_obs + 1
    coloring = fill(UInt8(0), n_total, n_total)
    for step in tree
        queue = copy(step.vertices)
        head = 1
        while head <= length(queue)
            u = queue[head]
            head += 1
            if coloring[u, u] != 0
                continue
            end
            coloring[u, u] = step.graph_id
            for v in 1:n_total
                if u == v || coloring[u, v] != 0
                    continue
                end
                coloring[u, v] = step.graph_id
                coloring[v, u] = step.graph_id
                if has_edge(condensation, n_obs, u, v)
                    push!(queue, v)
                end
            end
        end
    end
    return coloring
end

function apply_decision_tree_packed(n_obs::Int, condensation::CondensationBits, tree::DecisionTree)::PackedColoring
    return pack_coloring(apply_decision_tree_matrix(n_obs, condensation, tree))
end
