"""
    canonical_ray(r, config=OracleConfig()) -> Vector{Float64}

Normalize ray `r` by dividing by its infinity-norm. Returns the zero vector if
`‖r‖_∞ ≤ config.canonical_tol`.

See also: [`canonical_normal`](@ref), [`hash_key_ray`](@ref).
"""
function canonical_ray(r::AbstractVector{<:Real}, config::OracleConfig=OracleConfig())
    w = Float64.(collect(r))
    isempty(w) && return w
    s = maximum(abs, w)
    if s <= config.canonical_tol
        return zeros(Float64, length(w))
    end
    return w ./ s
end

"""
    canonical_normal(h, config=OracleConfig()) -> Vector{Float64}

Normalize facet normal `h` by dividing by its infinity-norm. Returns the zero vector if
`‖h‖_∞ ≤ config.canonical_tol`.

See also: [`canonical_ray`](@ref), [`hash_key_normal`](@ref).
"""
function canonical_normal(h::AbstractVector{<:Real}, config::OracleConfig=OracleConfig())
    w = Float64.(collect(h))
    isempty(w) && return w
    s = maximum(abs, w)
    if s <= config.canonical_tol
        return zeros(Float64, length(w))
    end
    return w ./ s
end

"""
    hash_key_ray(r, config=OracleConfig()) -> HashKey

Compute a quantized integer tuple from ray `r` for use in deduplication hash sets.
The ray is first canonicalized, then each coordinate is rounded to the nearest multiple of
`config.quantization_eps`.

See also: [`hash_key_normal`](@ref), [`canonical_ray`](@ref).
"""
function hash_key_ray(r::AbstractVector{<:Real}, config::OracleConfig=OracleConfig())
    w = canonical_ray(r, config)
    return Tuple(round.(Int, w ./ config.quantization_eps))
end

"""
    hash_key_normal(h, config=OracleConfig()) -> HashKey

Compute a quantized integer tuple from facet normal `h` for use in deduplication hash sets.
The normal is first canonicalized, then each coordinate is rounded to the nearest multiple of
`config.quantization_eps`.

See also: [`hash_key_ray`](@ref), [`canonical_normal`](@ref).
"""
function hash_key_normal(h::AbstractVector{<:Real}, config::OracleConfig=OracleConfig())
    w = canonical_normal(h, config)
    return Tuple(round.(Int, w ./ config.quantization_eps))
end
