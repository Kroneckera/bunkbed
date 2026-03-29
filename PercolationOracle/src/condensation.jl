export CondensationBits, encode_condensation, decode_condensation, has_edge
export condensation_edge_pairs, condensation_from_partition_id

const CondensationBits = UInt16

function condensation_edge_pairs(n::Int)
    1 <= n <= MAX_OBSERVABLES || throw(ArgumentError("supported n are 1:$MAX_OBSERVABLES, got $n"))
    pairs = Tuple{Int,Int}[]
    for u in 1:n-1, v in u+1:n
        push!(pairs, (u, v))
    end
    return pairs
end

const CONDENSATION_EDGE_PAIRS = Dict{Int,Vector{Tuple{Int,Int}}}(n => condensation_edge_pairs(n) for n in 1:MAX_OBSERVABLES)

function encode_condensation(n::Int, matrix::AbstractMatrix{Bool})::CondensationBits
    size(matrix, 1) == n + 1 || throw(ArgumentError("expected a $(n+1)x$(n+1) condensation matrix"))
    bits = zero(CondensationBits)
    for (idx, (u, v)) in enumerate(CONDENSATION_EDGE_PAIRS[n])
        if matrix[u, v] || matrix[v, u]
            bits |= CondensationBits(1) << (idx - 1)
        end
    end
    return bits
end

function decode_condensation(n::Int, bits::CondensationBits)
    matrix = zeros(Bool, n + 1, n + 1)
    for (idx, (u, v)) in enumerate(CONDENSATION_EDGE_PAIRS[n])
        if ((bits >> (idx - 1)) & CondensationBits(1)) == CondensationBits(1)
            matrix[u, v] = true
            matrix[v, u] = true
        end
    end
    return matrix
end

function _edge_bit_index(n::Int, u::Int, v::Int)
    a, b = u < v ? (u, v) : (v, u)
    # Bit index for edge (a, b) in the upper triangle of [1..n]:
    # pairs are ordered (1,2),(1,3),...,(1,n),(2,3),...,(n-1,n)
    return (a - 1) * n - a * (a - 1) ÷ 2 + (b - a)
end

function has_edge(bits::CondensationBits, n::Int, u::Int, v::Int)
    (u == v || u > n || v > n || u < 1 || v < 1) && return false
    idx = _edge_bit_index(n, u, v)
    return ((bits >> (idx - 1)) & CondensationBits(1)) == CondensationBits(1)
end

function condensation_from_partition_id(n::Int, id::Integer)::CondensationBits
    bits = zero(CondensationBits)
    for block in partition_blocks(n, id)
        length(block) <= 1 && continue
        for i in 1:length(block)-1, j in i+1:length(block)
            idx = _edge_bit_index(n, Int(block[i]), Int(block[j]))
            bits |= CondensationBits(1) << (idx - 1)
        end
    end
    return bits
end
