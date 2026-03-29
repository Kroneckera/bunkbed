export BLOCKED_PSI, TransitionTable, build_transition_table, next_psi_value, mask_value

const BLOCKED_PSI = typemax(UInt8)

struct TransitionTable
    n_obs::Int
    n_total::Int
    bell::Int
    num_colorings::Int
    next_psi::Vector{UInt8}
    mask_values::Vector{UInt8}
end

function transition_index(tt::TransitionTable, tree_idx::Int, partition_id::Integer, component::Int, next_component::Int, current_psi::Integer)
    bell_offset = Int(partition_id)
    psi_base = tt.n_obs + 1
    idx = (((tree_idx - 1) * tt.bell + bell_offset) * tt.n_total + (component - 1))
    idx = (idx * tt.n_total + (next_component - 1)) * psi_base + Int(current_psi)
    return idx + 1
end

function mask_index(tt::TransitionTable, tree_idx::Int, partition_id::Integer, vertex::Int)
    idx = ((tree_idx - 1) * tt.bell + Int(partition_id)) * tt.n_total + (vertex - 1)
    return idx + 1
end

function build_transition_table(n_obs::Int, colorings::AbstractVector{PackedColoring})
    n_total = n_obs + 1
    bell = bell_number(n_obs)
    num_colorings = length(colorings)
    next_psi = Vector{UInt8}(undef, num_colorings * bell * n_total * n_total * (n_obs + 1))
    mask_values = Vector{UInt8}(undef, num_colorings * bell * n_total)
    tt = TransitionTable(n_obs, n_total, bell, num_colorings, next_psi, mask_values)

    for tree_idx in 1:num_colorings
        coloring = colorings[tree_idx]
        for partition_id in 0:bell-1
            assignments = partition_assignments(n_obs, partition_id)
            for vertex in 1:n_total
                mask = UInt8(0)
                if vertex <= n_obs && coloring_value(coloring, n_obs, vertex, vertex) == UInt8(1)
                    mask = UInt8(assignments[vertex])
                end
                mask_values[mask_index(tt, tree_idx, partition_id, vertex)] = mask
            end
            for component in 1:n_total
                for next_component in 1:n_total
                    passive_color = coloring_value(coloring, n_obs, component, next_component)
                    mask = mask_values[mask_index(tt, tree_idx, partition_id, next_component)]
                    for current_psi in 0:n_obs
                        value = if passive_color == UInt8(1)
                            mask
                        elseif current_psi != 0 && mask != 0 && mask != current_psi
                            BLOCKED_PSI
                        else
                            UInt8(max(Int(mask), current_psi))
                        end
                        next_psi[transition_index(tt, tree_idx, partition_id, component, next_component, current_psi)] = value
                    end
                end
            end
        end
    end
    return tt
end

function next_psi_value(tt::TransitionTable, tree_idx::Int, partition_id::Integer, component::Int, next_component::Int, current_psi::Integer)::UInt8
    return tt.next_psi[transition_index(tt, tree_idx, partition_id, component, next_component, current_psi)]
end

function mask_value(tt::TransitionTable, tree_idx::Int, partition_id::Integer, vertex::Int)::UInt8
    return tt.mask_values[mask_index(tt, tree_idx, partition_id, vertex)]
end
