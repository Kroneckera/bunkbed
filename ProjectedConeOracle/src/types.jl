"""
Shared internal types for ProjectedConeOracle.
"""
const HashKey = Tuple{Vararg{Int}}

"""
    ProjectedConeResult

Result returned by [`projected_cone_oracle`](@ref) and [`projected_cone`](@ref).

# Fields
- `rays::Vector{Vector{Float64}}` — extreme rays of the projected cone `D = π_J(C)`.
- `facets::Vector{Vector{Float64}}` — facet normals (support hyperplanes) of `D`.
  Each facet `h` satisfies `h ⋅ y ≥ 0` for all `y ∈ D`.
- `iterations::Int` — number of separation–facet-enumeration iterations executed.
- `lp_calls::Int` — total number of LP solves (including two-phase verification).
- `converged::Bool` — `true` if the algorithm terminated with a complete description;
  `false` if `max_iterations` was reached.
"""
struct ProjectedConeResult
    rays::Vector{Vector{Float64}}
    facets::Vector{Vector{Float64}}
    iterations::Int
    lp_calls::Int
    converged::Bool
end

struct CheckpointPayload
    iter::Int
    lp_calls::Int
    rays::Vector{Vector{BigInt}}
    basis::Union{Nothing, Matrix{BigInt}}
end
