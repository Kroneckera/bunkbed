function _projected_dual_complement(n::Integer, J::AbstractVector{<:Integer})
    inJ = falses(n)
    for j in J
        1 <= j <= n || throw(ArgumentError("find_normalization_vector: J contains out-of-range index $j (n=$n)"))
        inJ[j] = true
    end
    K = Int[]
    sizehint!(K, n - length(J))
    for col in 1:n
        !inJ[col] && push!(K, col)
    end
    return K
end

function _build_normalization_model(
    A::SparseMatrixCSC{Float64,Int},
    J::Vector{Int},
    adapter::AbstractSolverAdapter,
)
    m, n = size(A)
    model = create_model(adapter)
    K = _projected_dual_complement(n, J)

    @variable(model, s[1:m] >= 0.0)
    @variable(model, tmin >= 0.0)
    @constraint(model, sum(s) + m * tmin == 1.0)

    for col in K
        start = A.colptr[col]
        stop = A.colptr[col + 1] - 1
        if start <= stop
            bcol = 0.0
            for ptr in start:stop
                bcol += A.nzval[ptr]
            end
            @constraint(model,
                sum(A.nzval[ptr] * s[A.rowval[ptr]] for ptr in start:stop) + bcol * tmin == 0.0,
            )
        end
    end

    @objective(model, Max, 1.0 * tmin)
    return model, s, tmin
end

function _fallback_lambda_qp(
    A::SparseMatrixCSC{Float64,Int},
    J::Vector{Int},
    adapter::AbstractSolverAdapter,
)
    m, n = size(A)
    K = _projected_dual_complement(n, J)
    model = create_model(adapter)
    @variable(model, λ[1:m] >= 0.0)
    @constraint(model, sum(λ) == 1.0)

    for col in K
        start = A.colptr[col]
        stop = A.colptr[col + 1] - 1
        if start <= stop
            @constraint(model, sum(A.nzval[ptr] * λ[A.rowval[ptr]] for ptr in start:stop) == 0.0)
        end
    end

    @objective(model, Min, sum(λ[i]^2 for i in 1:m))
    optimize!(model)
    status = termination_status(model)
    status == MOI.OPTIMAL || error("find_normalization_vector: fallback λ-QP failed with status: $status")
    return value.(λ)
end

function _compute_normalization_vector(A::SparseMatrixCSC{Float64,Int}, J::Vector{Int}, λ::Vector{Float64}, config::OracleConfig)
    c = zeros(Float64, length(J))
    for (t, col) in enumerate(J)
        start = A.colptr[col]
        stop = A.colptr[col + 1] - 1
        acc = 0.0
        for ptr in start:stop
            acc += A.nzval[ptr] * λ[A.rowval[ptr]]
        end
        c[t] = acc
    end
    c = canonical_normal(c, config)
    maximum(abs, c) > config.canonical_tol || error("find_normalization_vector: computed c is numerically zero")
    return c
end

function _seed_rays(
    A::SparseMatrixCSC{Float64,Int},
    J::Vector{Int},
    c::Vector{Float64},
    config::OracleConfig,
    adapter::AbstractSolverAdapter,
)
    oracle = SeparationOracle(A, J; adapter=adapter, c=c, config=config)
    seen = Set{HashKey}()
    seeds = Vector{Vector{Float64}}()
    d = length(J)
    for t in 1:d
        for sign in (-1.0, 1.0)
            h = zeros(Float64, d)
            h[t] = sign
            _, y, _ = solve_verified!(oracle, h; canonicalize=false, verify_tol=nothing)
            ray = canonical_ray(y, config)
            all(iszero, ray) && continue
            key = hash_key_ray(ray, config)
            if !(key in seen)
                push!(seen, key)
                push!(seeds, ray)
            end
        end
    end
    isempty(seeds) && error("find_normalization_vector: failed to generate initial seed rays")
    return seeds
end

"""
    find_normalization_vector(A, J, config=OracleConfig(), adapter=HiGHSAdapter()) -> (c, seeds)

Compute a normalization vector `c ∈ relint(D*)` for the projected cone `D = π_J({x : Ax ≥ 0})`
and an initial set of seed rays.

The normalization vector satisfies `c'y > 0` for all nonzero `y ∈ D`, enabling the
separation oracle to work on the compact slice `{y ∈ D : c'y = 1}`.

# Algorithm
1. Solve a max-min LP to find `λ ≥ 0` with `A'_K λ = 0` and `λ` as uniform as possible.
2. Compute `c = A'_J λ` (projected to `J`-coordinates).
3. Probe `±e_t` directions to generate initial seed rays.

Returns `(c, seeds)` where `c::Vector{Float64}` and `seeds::Vector{Vector{Float64}}`.

See also: [`find_dual_normalization_vector`](@ref), [`SeparationOracle`](@ref).
"""
function find_normalization_vector(
    A::SparseMatrixCSC{Float64,Int},
    J::Vector{Int},
    config::OracleConfig=OracleConfig(),
    adapter::AbstractSolverAdapter=HiGHSAdapter(),
)
    isempty(J) && throw(ArgumentError("find_normalization_vector: J must be non-empty"))
    model, s, tmin = _build_normalization_model(A, collect(J), adapter)
    optimize!(model)

    status = termination_status(model)
    status == MOI.OPTIMAL || error("find_normalization_vector: max-min LP failed with status: $status")

    tval = value(tmin)
    λ = value.(s) .+ tval
    if tval <= config.canonical_tol
        λ = _fallback_lambda_qp(A, collect(J), adapter)
    end
    c = _compute_normalization_vector(A, collect(J), λ, config)
    seeds = _seed_rays(A, collect(J), c, config, adapter)
    return c, seeds
end

"""
    find_dual_normalization_vector(A, J; adapter=HiGHSAdapter(), config=OracleConfig()) -> (c, seeds)

Keyword-argument wrapper for [`find_normalization_vector`](@ref). Provided for backward
compatibility with earlier API conventions.

See also: [`find_normalization_vector`](@ref).
"""
function find_dual_normalization_vector(
    A::SparseMatrixCSC{Float64,Int},
    J::Vector{Int};
    adapter::AbstractSolverAdapter=HiGHSAdapter(),
    config::OracleConfig=OracleConfig(),
    kwargs...,
)
    isempty(kwargs) || @warn "find_dual_normalization_vector: ignoring unsupported keyword arguments" kwargs=kwargs
    return find_normalization_vector(A, J, config, adapter)
end
