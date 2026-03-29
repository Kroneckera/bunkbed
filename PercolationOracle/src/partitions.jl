using Combinatorics

export MAX_OBSERVABLES, PartitionID, bell_number, all_partition_ids
export canonical_partition_label, encode_partition, decode_partition, partition_label
export partition_assignments, partition_blocks, partition_roots, partition_block_count

const MAX_OBSERVABLES = 4
const PartitionID = UInt8

struct PartitionTable
    partitions::Vector{Vector{Vector{Int}}}
    labels::Vector{String}
    lookup::Dict{String,PartitionID}
    assignments::Matrix{UInt8}
    block_counts::Vector{UInt8}
    blocks::Vector{Vector{Vector{UInt8}}}
    roots::Vector{Vector{UInt8}}
end

function canonicalize_partition_blocks(partition)
    blocks = Vector{Vector{Int}}()
    for block in partition
        values = sort!([Int(x) for x in block])
        isempty(values) && continue
        push!(blocks, values)
    end
    sort!(blocks, by = first)
    return blocks
end

function canonical_partition_label(blocks_or_partition)
    blocks = canonicalize_partition_blocks(blocks_or_partition)
    return join((join(string.(block)) for block in blocks), "|")
end

function build_partition_table(n::Int)
    raw_partitions = collect(partitions(1:n))
    bell = length(raw_partitions)

    partitions_store = Vector{Vector{Vector{Int}}}(undef, bell)
    labels = Vector{String}(undef, bell)
    lookup = Dict{String,PartitionID}()
    assignments = zeros(UInt8, bell, n)
    block_counts = Vector{UInt8}(undef, bell)
    blocks_store = Vector{Vector{Vector{UInt8}}}(undef, bell)
    roots_store = Vector{Vector{UInt8}}(undef, bell)

    for (idx, part) in enumerate(raw_partitions)
        blocks = canonicalize_partition_blocks(part)
        partitions_store[idx] = [copy(block) for block in blocks]
        labels[idx] = join((join(string.(block)) for block in blocks), "|")
        lookup[labels[idx]] = PartitionID(idx - 1)
        block_counts[idx] = UInt8(length(blocks))
        block_vectors = [UInt8.(block) for block in blocks]
        blocks_store[idx] = block_vectors
        roots_store[idx] = UInt8[first(block) for block in block_vectors]
        for (block_id, block) in enumerate(blocks)
            for vertex in block
                assignments[idx, vertex] = UInt8(block_id)
            end
        end
    end

    return PartitionTable(
        partitions_store,
        labels,
        lookup,
        assignments,
        block_counts,
        blocks_store,
        roots_store,
    )
end

const PARTITION_TABLES = Dict{Int,PartitionTable}(n => build_partition_table(n) for n in 1:MAX_OBSERVABLES)

partition_table(n::Int) = get(PARTITION_TABLES, n) do
    throw(ArgumentError("supported n are 1:$MAX_OBSERVABLES, got $n"))
end

bell_number(n::Int) = length(partition_table(n).partitions)
all_partition_ids(n::Int) = PartitionID.(0:bell_number(n)-1)

function partition_label(n::Int, id::Integer)
    table = partition_table(n)
    idx = Int(id) + 1
    1 <= idx <= length(table.labels) || throw(BoundsError(table.labels, idx))
    return table.labels[idx]
end

function encode_partition(n::Int, partition)::PartitionID
    table = partition_table(n)
    key = canonical_partition_label(partition)
    return get(table.lookup, key) do
        throw(ArgumentError("partition $key is not a partition of 1:$n"))
    end
end

function decode_partition(n::Int, id::Integer)
    table = partition_table(n)
    idx = Int(id) + 1
    1 <= idx <= length(table.partitions) || throw(BoundsError(table.partitions, idx))
    return [copy(block) for block in table.partitions[idx]]
end

function partition_assignments(n::Int, id::Integer)
    table = partition_table(n)
    idx = Int(id) + 1
    return @view table.assignments[idx, :]
end

function partition_blocks(n::Int, id::Integer)
    table = partition_table(n)
    idx = Int(id) + 1
    return table.blocks[idx]
end

function partition_roots(n::Int, id::Integer)
    table = partition_table(n)
    idx = Int(id) + 1
    return table.roots[idx]
end

function partition_block_count(n::Int, id::Integer)
    table = partition_table(n)
    idx = Int(id) + 1
    return Int(table.block_counts[idx])
end
