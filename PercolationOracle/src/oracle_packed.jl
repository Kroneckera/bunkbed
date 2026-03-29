export PackedOracleContext, OracleWorkspace
export check_g1_compatibility_packed, check_partition_compatibility_packed

struct PackedOracleContext
    n_obs::Int
    n_total::Int
    condensation::CondensationBits
    colorings::Vector{PackedColoring}
    transition_table::TransitionTable
end

function PackedOracleContext(n_obs::Int, condensation::CondensationBits, colorings::AbstractVector{PackedColoring})
    tt = build_transition_table(n_obs, colorings)
    return PackedOracleContext(n_obs, n_obs + 1, condensation, collect(colorings), tt)
end

mutable struct OracleWorkspace
    visited::BitVector
    queue::Vector{UInt64}
    radices::Vector{Int}
    strides::Vector{Int}
    sel_matrix::Matrix{UInt8}
    target_hits::Vector{Bool}
end

function OracleWorkspace(ctx::PackedOracleContext)
    return OracleWorkspace(
        falses(0),
        UInt64[],
        Int[],
        Int[],
        zeros(UInt8, ctx.n_total, max(length(ctx.colorings), 1)),
        falses(max(ctx.n_obs, 1)),
    )
end

function normalize_partition_ids(partition_ids::AbstractVector{<:Integer})
    return PartitionID[Int(pid) for pid in partition_ids]
end

function prepare_workspace!(workspace::OracleWorkspace, ctx::PackedOracleContext, partition_ids::AbstractVector{PartitionID})
    m = length(partition_ids)
    resize!(workspace.radices, m)
    resize!(workspace.strides, m)

    stride = ctx.n_total
    for k in 1:m
        pid = partition_ids[k]
        assignments = partition_assignments(ctx.n_obs, pid)
        for vertex in 1:ctx.n_obs
            workspace.sel_matrix[vertex, k] = UInt8(assignments[vertex])
        end
        workspace.sel_matrix[ctx.n_total, k] = UInt8(0)
        workspace.radices[k] = partition_block_count(ctx.n_obs, pid) + 1
        workspace.strides[k] = stride
        stride *= workspace.radices[k]
    end

    resize!(workspace.visited, stride)
    fill!(workspace.visited, false)
    if length(workspace.queue) < stride
        resize!(workspace.queue, stride)
    end
    return stride
end

function pack_selected_state(component::Int, sel_matrix::Matrix{UInt8}, m::Int)
    T = state_word_type(m)
    bits = T(component)
    @inbounds for k in 1:m
        bits |= (T(sel_matrix[component, k]) << (3 * k))
    end
    return UInt64(bits)
end

function build_star_loop_state(ctx::PackedOracleContext, state::UInt64, active_index::Int, m::Int)
    T = state_word_type(m)
    bits = T(ctx.n_total)
    active_diag = coloring_value(ctx.colorings[active_index], ctx.n_obs, ctx.n_total, ctx.n_total)
    @inbounds for k in 1:m
        psi_value = coloring_value(ctx.colorings[k], ctx.n_obs, ctx.n_total, ctx.n_total) == active_diag ? psi_from_state(UInt64(state), k) : UInt8(0)
        bits |= (T(psi_value) << (3 * k))
    end
    return UInt64(bits)
end

function build_nonstar_state(ctx::PackedOracleContext, partition_ids::AbstractVector{PartitionID}, state::UInt64, component::Int, next_component::Int, active_index::Int, m::Int)
    T = state_word_type(m)
    active_color = coloring_value(ctx.colorings[active_index], ctx.n_obs, component, next_component)

    if active_color == UInt8(1)
        allowed = (component != next_component && has_edge(ctx.condensation, ctx.n_obs, component, next_component)) ||
                  (component == next_component && component != ctx.n_total)
        allowed || return 0x00, false
    end

    bits = T(next_component)
    tt = ctx.transition_table
    @inbounds for k in 1:m
        current_psi = active_color == UInt8(1) ? 0 : Int(psi_from_state(UInt64(state), k))
        next_psi = next_psi_value(tt, k, partition_ids[k], component, next_component, current_psi)
        next_psi == BLOCKED_PSI && return 0x00, false
        bits |= (T(next_psi) << (3 * k))
    end
    return UInt64(bits), true
end

function compatible_with_vertex(state::UInt64, vertex::Int, sel_matrix::Matrix{UInt8}, m::Int)
    @inbounds for k in 1:m
        psi_value = psi_from_state(UInt64(state), k)
        if psi_value != 0 && psi_value != sel_matrix[vertex, k]
            return false
        end
    end
    return true
end

function check_g1_compatibility_packed(ctx::PackedOracleContext, partition_ids::AbstractVector{PartitionID})
    m = length(partition_ids)
    @inbounds for k in 1:m
        assignments = partition_assignments(ctx.n_obs, partition_ids[k])
        coloring = ctx.colorings[k]
        for u in 1:ctx.n_obs-1
            for v in u+1:ctx.n_obs
                if assignments[u] != assignments[v] && coloring_value(coloring, ctx.n_obs, u, v) == UInt8(1) && has_edge(ctx.condensation, ctx.n_obs, u, v)
                    return false
                end
            end
        end
    end
    return true
end

function check_partition_compatibility_packed(ctx::PackedOracleContext, partition_ids::AbstractVector{<:Integer}; workspace::Union{Nothing,OracleWorkspace}=nothing)
    isempty(partition_ids) && return true
    normalized = partition_ids isa AbstractVector{PartitionID} ? partition_ids : normalize_partition_ids(partition_ids)
    local_workspace = isnothing(workspace) ? OracleWorkspace(ctx) : workspace
    prepare_workspace!(local_workspace, ctx, normalized)

    check_g1_compatibility_packed(ctx, normalized) || return false

    m = length(normalized)
    @inbounds for active_index in 1:m
        for block in partition_blocks(ctx.n_obs, normalized[active_index])
            block_length = length(block)
            if block_length <= 1
                continue
            end

            fill!(local_workspace.target_hits, false)
            local_workspace.target_hits[1] = true
            remaining = block_length - 1

            fill!(local_workspace.visited, false)
            head = 1
            tail = 1
            start_state = pack_selected_state(Int(block[1]), local_workspace.sel_matrix, m)
            local_workspace.queue[1] = start_state
            local_workspace.visited[state_index(UInt64(start_state), ctx.n_total, local_workspace.radices, local_workspace.strides)] = true

            while head <= tail && remaining > 0
                state = local_workspace.queue[head]
                head += 1
                component = component_from_state(UInt64(state))
                for next_component in 1:ctx.n_total
                    next_state, ok = if component == ctx.n_total && next_component == ctx.n_total
                        build_star_loop_state(ctx, state, active_index, m), true
                    else
                        build_nonstar_state(ctx, normalized, state, component, next_component, active_index, m)
                    end
                    ok || continue

                    idx = state_index(UInt64(next_state), ctx.n_total, local_workspace.radices, local_workspace.strides)
                    if local_workspace.visited[idx]
                        continue
                    end
                    local_workspace.visited[idx] = true
                    tail += 1
                    local_workspace.queue[tail] = next_state

                    if next_component <= ctx.n_obs
                        for (local_idx, vertex_u8) in enumerate(block)
                            if local_workspace.target_hits[local_idx]
                                continue
                            end
                            vertex = Int(vertex_u8)
                            if next_component == vertex && compatible_with_vertex(next_state, vertex, local_workspace.sel_matrix, m)
                                local_workspace.target_hits[local_idx] = true
                                remaining -= 1
                                remaining == 0 && break
                            end
                        end
                    end
                end
            end

            remaining == 0 || return false
        end
    end

    return true
end
