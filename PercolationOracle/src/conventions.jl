export PAPER_J3_ORDER_IDS, default_partition_order, partition_order_ids, partition_order_labels
export reorder_partition_matrix, normalize_feasible_tuple, normalize_archive_feasible_tuple

const PAPER_J3_ORDER_IDS = PartitionID[0, 4, 3, 1, 2]

function default_partition_order(n_obs::Int)
    return n_obs == 3 ? :paper_n3 : :standard
end

function _normalize_partition_order(n_obs::Int, order::Symbol)
    if order in (:standard, :baseline, :internal)
        return :standard
    elseif order in (:paper, :paper_n3, :appendixA)
        n_obs == 3 || throw(ArgumentError("paper order is only defined for n_obs=3"))
        return :paper_n3
    else
        throw(ArgumentError("unknown partition order: $order"))
    end
end

function partition_order_ids(n_obs::Int; order::Symbol=default_partition_order(n_obs))
    normalized = _normalize_partition_order(n_obs, order)
    if normalized === :standard
        return collect(all_partition_ids(n_obs))
    end
    return collect(PAPER_J3_ORDER_IDS)
end

function partition_order_labels(n_obs::Int; order::Symbol=default_partition_order(n_obs))
    return [partition_label(n_obs, pid) for pid in partition_order_ids(n_obs; order=order)]
end

function reorder_partition_matrix(
    matrix::AbstractMatrix,
    n_obs::Int;
    from_order::Symbol=:standard,
    to_order::Symbol=default_partition_order(n_obs),
)
    size(matrix, 1) == size(matrix, 2) || throw(ArgumentError("pair matrix must be square"))
    from_ids = partition_order_ids(n_obs; order=from_order)
    to_ids = partition_order_ids(n_obs; order=to_order)
    size(matrix, 1) == length(from_ids) || throw(ArgumentError("matrix size $(size(matrix)) incompatible with n_obs=$n_obs and from_order=$from_order"))
    lookup = Dict(id => pos for (pos, id) in enumerate(from_ids))
    perm = [lookup[id] for id in to_ids]
    return matrix[perm, perm]
end

_archive_pairs_to_paper_pairs(pairs::NTuple{4,<:Any}) = (pairs[1], pairs[3], pairs[2], pairs[4])

function _flatten_tuple_values(tuple_like)
    values = Int[]
    if tuple_like isa AbstractVector{<:Integer} || (tuple_like isa Tuple && all(item -> item isa Integer, tuple_like))
        append!(values, Int.(tuple_like))
        return values
    end
    for item in tuple_like
        if !(item isa Tuple || item isa AbstractVector)
            throw(ArgumentError("unsupported feasible-tuple item type: $(typeof(item))"))
        end
        length(item) == 2 || throw(ArgumentError("paired feasible tuple items must have length 2, got length $(length(item))"))
        append!(values, Int.(item))
    end
    return values
end

function _detect_partition_id_base(values::AbstractVector{<:Integer}, bell::Int)
    isempty(values) && throw(ArgumentError("cannot detect partition-id base from empty values"))
    ints = Int.(values)
    minv = minimum(ints)
    maxv = maximum(ints)
    if any(==(0), ints)
        return :zero_based
    elseif any(==(bell), ints)
        return :one_based
    elseif 1 <= minv && maxv <= bell
        return :one_based
    elseif 0 <= minv && maxv < bell
        return :zero_based
    else
        throw(ArgumentError("partition ids outside valid ranges for Bell number $bell: min=$minv max=$maxv"))
    end
end

function _detect_partition_id_base(feasible_tuples, bell::Int)
    values = Int[]
    for tuple_like in feasible_tuples
        append!(values, _flatten_tuple_values(tuple_like))
    end
    return _detect_partition_id_base(values, bell)
end

function _normalize_partition_id(id::Integer, bell::Int; id_base::Symbol=:auto)
    base = id_base === :auto ? _detect_partition_id_base([Int(id)], bell) : id_base
    normalized = if base === :zero_based
        Int(id)
    elseif base === :one_based
        Int(id) - 1
    else
        throw(ArgumentError("unsupported id_base: $id_base"))
    end
    0 <= normalized < bell || throw(ArgumentError("normalized partition id $normalized outside 0:$(bell - 1)"))
    return PartitionID(normalized)
end

function _normalize_partition_ids(values::AbstractVector{<:Integer}, bell::Int; id_base::Symbol=:auto)
    base = id_base === :auto ? _detect_partition_id_base(values, bell) : id_base
    return PartitionID[_normalize_partition_id(value, bell; id_base=base) for value in values]
end

function normalize_feasible_tuple(
    tuple_like,
    n_obs::Int,
    m::Int;
    id_base::Symbol=:auto,
)
    bell = bell_number(n_obs)
    if !(tuple_like isa AbstractVector{<:Integer} || (tuple_like isa Tuple && all(item -> item isa Integer, tuple_like)))
        paired = collect(tuple_like)
        length(paired) == m || throw(ArgumentError("expected $m pair items, got $(length(paired))"))
        pairs = ntuple(k -> begin
            item = paired[k]
            length(item) == 2 || throw(ArgumentError("paired feasible tuple items must have length 2, got length $(length(item))"))
            ids = _normalize_partition_ids(Int[item[1], item[2]], bell; id_base=id_base)
            (ids[1], ids[2])
        end, m)
        return pairs
    end

    values = Int.(tuple_like)
    length(values) == 2m || throw(ArgumentError("expected length $(2m) feasible tuple, got length $(length(values))"))
    ids = _normalize_partition_ids(values, bell; id_base=id_base)
    return ntuple(k -> (ids[2k - 1], ids[2k]), m)
end

function normalize_archive_feasible_tuple(tuple_like, n_obs::Int=3; id_base::Symbol=:auto)
    n_obs == 3 || throw(ArgumentError("legacy archive tuple normalization is only defined for n_obs=3"))
    return _archive_pairs_to_paper_pairs(normalize_feasible_tuple(tuple_like, n_obs, 4; id_base=id_base))
end
