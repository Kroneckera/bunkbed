"""
    OracleConfig(; kwargs...)

Configuration for the projected cone oracle algorithm.  Every field has a
validated default derived from `base_eps`.

# Key parameters
- `base_eps::Float64 = 1e-9` — master tolerance; most other tolerances scale from this.
- `max_iterations::Int` (alias `max_rounds`) — iteration limit (default 10 000).
- `checkpoint_file::Union{String,Nothing} = nothing` — path for resumable checkpoints.
- `verbose::Bool = false` — print iteration-level progress to `log_io`.

# Tolerance hierarchy (all derived from `base_eps`)
- `canonical_tol = base_eps * 1e-3` — near-zero threshold for canonicalization.
- `quantization_eps = base_eps * 10` — hash-key rounding for ray/facet deduplication.
- `violation_tol = base_eps * 1000` — LP separation threshold.
- `seed_tol = violation_tol` — tolerance for deterministic seeding / subspace expansion.

# Rationalization
- `rat_tol::Float64 = 1e-10` — initial tolerance for `rationalize`.
- `max_den::Int = 10_000_000` — maximum denominator for rational approximation.
- `rat_tol_ceiling::Float64 = 1e-4` — maximum tolerance after doubling.

# Normaliz backend
- `normaliz_threads::Int = 0` — thread count for Normaliz (0 = auto).
- `normaliz_algorithm::Symbol = :projection_float` — one of `:none`, `:projection`,
  `:projection_float`.

# Example
```julia
config = OracleConfig(base_eps=1e-8, max_iterations=5000, verbose=true)
```
"""
struct OracleConfig
    base_eps::Float64
    canonical_tol::Float64
    quantization_eps::Float64
    pointedness_tol::Float64
    homogeneity_tol::Float64
    violation_tol::Float64
    seed_tol::Float64
    cy_normalize_tol::Float64
    rat_tol::Float64
    max_den::Int
    rat_tol_ceiling::Float64
    max_rounds::Int
    prune_every::Int
    shuffle_facets::Bool
    normaliz_threads::Int
    normaliz_verbose::Bool
    normaliz_keep_order::Bool
    normaliz_algorithm::Symbol
    checkpoint_file::Union{String,Nothing}
    checkpoint_reorthogonalize::Bool
    verbose::Bool
    log_io::IO
end

function OracleConfig(; 
    base_eps::Float64 = 1e-9,
    canonical_tol::Float64 = base_eps * 1e-3,
    quantization_eps::Float64 = base_eps * 10,
    pointedness_tol::Float64 = base_eps * 10,
    homogeneity_tol::Float64 = base_eps * 100,
    violation_tol::Float64 = base_eps * 1000,
    seed_tol::Float64 = violation_tol,
    cy_normalize_tol::Float64 = 1e-12,
    rat_tol::Float64 = 1e-10,
    max_den::Int = 10_000_000,
    rat_tol_ceiling::Float64 = 1e-4,
    max_rounds::Union{Nothing,Int} = nothing,
    max_iterations::Union{Nothing,Int} = nothing,
    prune_every::Int = 100,
    shuffle_facets::Bool = true,
    normaliz_threads::Int = 0,
    normaliz_verbose::Bool = false,
    normaliz_keep_order::Bool = true,
    normaliz_algorithm::Symbol = :projection_float,
    checkpoint_file::Union{String,Nothing} = nothing,
    checkpoint_reorthogonalize::Bool = true,
    verbose::Bool = false,
    log_io::IO = stderr,
)
    max_rounds_eff = if max_rounds === nothing && max_iterations === nothing
        10_000
    elseif max_rounds === nothing
        max_iterations
    elseif max_iterations === nothing
        max_rounds
    elseif max_rounds == max_iterations
        max_rounds
    else
        throw(ArgumentError("max_rounds and max_iterations must agree when both are provided"))
    end

    base_eps > 0 || throw(ArgumentError("base_eps must be positive"))
    canonical_tol > 0 || throw(ArgumentError("canonical_tol must be positive"))
    quantization_eps > 0 || throw(ArgumentError("quantization_eps must be positive"))
    pointedness_tol > 0 || throw(ArgumentError("pointedness_tol must be positive"))
    homogeneity_tol > 0 || throw(ArgumentError("homogeneity_tol must be positive"))
    violation_tol > 0 || throw(ArgumentError("violation_tol must be positive"))
    seed_tol > 0 || throw(ArgumentError("seed_tol must be positive"))
    seed_tol >= violation_tol || throw(ArgumentError("seed_tol must be >= violation_tol"))
    cy_normalize_tol > 0 || throw(ArgumentError("cy_normalize_tol must be positive"))
    rat_tol > 0 || throw(ArgumentError("rat_tol must be positive"))
    max_den > 0 || throw(ArgumentError("max_den must be positive"))
    rat_tol_ceiling > rat_tol || throw(ArgumentError("rat_tol_ceiling must be > rat_tol"))
    max_rounds_eff > 0 || throw(ArgumentError("max_rounds must be positive"))
    prune_every >= 0 || throw(ArgumentError("prune_every must be >= 0"))
    normaliz_threads >= 0 || throw(ArgumentError("normaliz_threads must be >= 0"))
    normaliz_algorithm in (:none, :projection, :projection_float) ||
        throw(ArgumentError("normaliz_algorithm must be :none, :projection, or :projection_float"))

    return OracleConfig(
        base_eps,
        canonical_tol,
        quantization_eps,
        pointedness_tol,
        homogeneity_tol,
        violation_tol,
        seed_tol,
        cy_normalize_tol,
        rat_tol,
        max_den,
        rat_tol_ceiling,
        max_rounds_eff,
        prune_every,
        shuffle_facets,
        normaliz_threads,
        normaliz_verbose,
        normaliz_keep_order,
        normaliz_algorithm,
        checkpoint_file,
        checkpoint_reorthogonalize,
        verbose,
        log_io,
    )
end

effective_normaliz_threads(config::OracleConfig) =
    config.normaliz_threads == 0 ? Sys.CPU_THREADS : config.normaliz_threads

function Base.getproperty(config::OracleConfig, name::Symbol)
    if name === :max_iterations
        return getfield(config, :max_rounds)
    end
    return getfield(config, name)
end

function Base.propertynames(::OracleConfig, private::Bool=false)
    names = fieldnames(OracleConfig)
    return private ? (names..., :max_iterations) : (names..., :max_iterations)
end
