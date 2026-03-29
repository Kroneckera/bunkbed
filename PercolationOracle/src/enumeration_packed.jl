export enumerate_compatible_partitions_packed

function enumerate_compatible_partitions_packed(ctx::PackedOracleContext, prefix::AbstractVector{<:Integer}=PartitionID[])
    current_prefix = normalize_partition_ids(prefix)
    compatible = Vector{Vector{PartitionID}}()
    init_length = length(current_prefix)
    D = bell_number(ctx.n_obs)
    last_id = PartitionID(D - 1)
    workspace = OracleWorkspace(ctx)

    while true
        status = isempty(current_prefix) || check_partition_compatibility_packed(ctx, current_prefix; workspace=workspace)
        if status && length(current_prefix) < length(ctx.colorings)
            push!(current_prefix, PartitionID(0))
        else
            if status
                push!(compatible, copy(current_prefix))
            end
            while length(current_prefix) > init_length && current_prefix[end] == last_id
                pop!(current_prefix)
            end
            if length(current_prefix) == init_length
                break
            end
            current_prefix[end] = PartitionID(Int(current_prefix[end]) + 1)
        end
    end

    return compatible
end
