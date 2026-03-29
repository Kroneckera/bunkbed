export PackedState, state_word_type, packed_state_bits
export pack_state, unpack_state, component_from_state, psi_from_state
export state_index, mixed_radix_strides, state_count

const PackedState = Union{UInt32, UInt64}

state_word_type(num_colorings::Int) = (3 + 3 * num_colorings <= 32) ? UInt32 : UInt64
packed_state_bits(num_colorings::Int) = 3 + 3 * num_colorings

function pack_state(component::Integer, psi_values::AbstractVector{<:Integer})::PackedState
    T = state_word_type(length(psi_values))
    bits = T(component)
    for (idx, psi_value) in enumerate(psi_values)
        bits |= (T(psi_value) << (3 * idx))
    end
    return bits
end

component_from_state(state::PackedState) = Int(UInt64(state) & 0x7)
psi_from_state(state::PackedState, idx::Int) = UInt8((UInt64(state) >> (3 * idx)) & 0x7)

function unpack_state(state::PackedState, num_colorings::Int)
    psi_values = Vector{UInt8}(undef, num_colorings)
    for idx in 1:num_colorings
        psi_values[idx] = psi_from_state(state, idx)
    end
    return component_from_state(state), psi_values
end

function mixed_radix_strides(n_total::Int, radices::AbstractVector{<:Integer})
    strides = Vector{Int}(undef, length(radices))
    stride = n_total
    for idx in eachindex(radices)
        strides[idx] = stride
        stride *= Int(radices[idx])
    end
    return strides
end

function state_count(n_total::Int, radices::AbstractVector{<:Integer})
    total = n_total
    for radix in radices
        total *= Int(radix)
    end
    return total
end

function state_index(state::PackedState, n_total::Int, radices::AbstractVector{<:Integer}, strides::AbstractVector{<:Integer})
    idx = component_from_state(state)
    @inbounds for k in eachindex(radices)
        idx += Int(psi_from_state(state, k)) * Int(strides[k])
    end
    return idx
end
