function _algorithm_log(config::OracleConfig, message::AbstractString)
    config.verbose && println(config.log_io, message)
    return nothing
end

function _initial_subspace_basis(rays::Vector{Vector{Float64}}, d::Integer, config::OracleConfig)
    sb = nothing
    for ray in rays
        v = Vector{Float64}(ray)
        nv = norm(v)
        nv > config.canonical_tol || continue
        v ./= nv
        if sb === nothing
            sb = SubspaceBasis(reshape(v, Int(d), 1))
        else
            maybe_expand!(sb, v, config)
        end
    end
    sb === nothing && error("projected_cone_oracle: unable to initialize a subspace basis from the available rays")
    return sb
end

function _reduced_normalization_vector(sb::SubspaceBasis, c_y::AbstractVector{<:Real}, config::OracleConfig)
    c_z = project(sb, c_y)
    maximum(abs, c_z) > config.canonical_tol ||
        error("projected_cone_oracle: projected normalization vector is numerically zero in the current subspace")
    return c_z
end

function _recompute_reduced_rays(
    rays_y::Vector{Vector{Float64}},
    sb::SubspaceBasis,
    c_y::AbstractVector{<:Real},
    config::OracleConfig,
)
    c_z = _reduced_normalization_vector(sb, c_y, config)
    rays_z = Vector{Vector{Float64}}(undef, length(rays_y))
    for (idx, ray_y) in enumerate(rays_y)
        rays_z[idx] = cy_normalize(project(sb, ray_y), c_z, config)
    end
    seen = Set{HashKey}(hash_key_ray(ray_z, config) for ray_z in rays_z)
    return c_z, rays_z, seen
end

function _rationalized_ray_matrix(rays_z::Vector{Vector{Float64}}, config::OracleConfig)
    isempty(rays_z) && throw(ArgumentError("_rationalized_ray_matrix: at least one ray is required"))
    out = Matrix{BigInt}(undef, length(rays_z), length(first(rays_z)))
    for (idx, ray_z) in enumerate(rays_z)
        out[idx, :] = rationalize_ray_coordwise(ray_z, config)
    end
    return out
end

function _dedupe_vectors(
    vectors::Vector{Vector{Float64}},
    hashfn::Function,
    config::OracleConfig,
)
    seen = Set{HashKey}()
    unique_vectors = Vector{Vector{Float64}}()
    for vec in vectors
        key = hashfn(vec, config)
        if !(key in seen)
            push!(seen, key)
            push!(unique_vectors, vec)
        end
    end
    return unique_vectors
end

function _lift_rays(M::AbstractMatrix{<:Integer}, sb::SubspaceBasis, config::OracleConfig)
    rays = Vector{Vector{Float64}}(undef, size(M, 1))
    for i in 1:size(M, 1)
        rays[i] = canonical_ray(lift(sb, Float64.(collect(M[i, :]))), config)
    end
    return _dedupe_vectors(rays, hash_key_ray, config)
end

function _lift_facets(M::AbstractMatrix{<:Integer}, sb::SubspaceBasis, config::OracleConfig)
    facets = Vector{Vector{Float64}}(undef, size(M, 1))
    for i in 1:size(M, 1)
        facets[i] = canonical_normal(lift(sb, Float64.(collect(M[i, :]))), config)
    end
    return _dedupe_vectors(facets, hash_key_normal, config)
end

function _result_from_current_cone(
    rays_z::Vector{Vector{Float64}},
    sb::SubspaceBasis,
    iter::Int,
    lp_calls::Int,
    config::OracleConfig,
    backend::AbstractNormalizBackend,
    converged::Bool,
)
    G = _rationalized_ray_matrix(rays_z, config)
    result = compute_cone(backend, G, config)
    is_pointed(result) || error(
        "projected_cone_oracle: current inner approximation has lineality space of dimension $(size(result.lineality_space, 1))"
    )
    rays = _lift_rays(result.extreme_rays, sb, config)
    facets = _lift_facets(result.support_hyperplanes, sb, config)
    return ProjectedConeResult(rays, facets, iter, lp_calls, converged)
end

"""
    projected_cone_oracle(A, J, config, adapter; backend=NormalizCLIBackend())
    projected_cone_oracle(A, J; config, solver, backend)

Compute the projected cone `D = π_J(C)` where `C = {x : Ax ≥ 0}` and `π_J`
projects onto the coordinate indices in `J`.

Uses an oracle-driven double-description algorithm: iteratively solve LP
separation problems to discover extreme rays, then call Normaliz to enumerate
facets of the current inner approximation, until no violated facet remains.

# Arguments
- `A` — constraint matrix defining the cone `C = {x ∈ ℝⁿ : Ax ≥ 0}`.
- `J` — vector of projection indices (1-based), `length(J) = d`.
- `config::OracleConfig` — algorithm configuration (tolerances, iteration limit, etc.).
- `adapter::AbstractSolverAdapter` — LP solver backend (`HiGHSAdapter()`, `GurobiAdapter()`, etc.).
- `backend::AbstractNormalizBackend` — Normaliz backend for facet enumeration (default: CLI).

# Returns
A [`ProjectedConeResult`](@ref) with fields `rays`, `facets`, `iterations`,
`lp_calls`, and `converged`.

# Example
```julia
using ProjectedConeOracle, SparseArrays
A = sparse(Float64[1 0; 0 1])  # positive orthant
result = projected_cone_oracle(A, [1, 2])
result.rays   # [[0.0, 1.0], [1.0, 0.0]]
result.facets # [[0.0, 1.0], [1.0, 0.0]]
```
"""
function projected_cone_oracle(
    A::AbstractMatrix{<:Real},
    J::AbstractVector{<:Integer},
    config::OracleConfig,
    adapter::AbstractSolverAdapter;
    backend::AbstractNormalizBackend=NormalizCLIBackend(),
)
    Jvec = Int.(collect(J))
    isempty(Jvec) && throw(ArgumentError("projected_cone_oracle: J must be non-empty"))

    A_sparse = sparse(Float64.(A))
    d = length(Jvec)
    c_y, seed_rays = find_normalization_vector(A_sparse, Jvec, config, adapter)

    rays_y = Vector{Vector{Float64}}()
    iter = 0
    lp_calls = 0
    resumed = false

    sb = if config.checkpoint_file !== nothing && isfile(config.checkpoint_file)
        rays_y, _, iter, lp_calls, B_loaded = load_checkpoint(config.checkpoint_file, config)
        resumed = true
        isempty(rays_y) && append!(rays_y, seed_rays)
        B_loaded === nothing ? SubspaceBasis(Matrix{Float64}(I, d, d)) : SubspaceBasis(B_loaded)
    else
        append!(rays_y, seed_rays)
        _initial_subspace_basis(rays_y, d, config)
    end

    isempty(rays_y) && append!(rays_y, seed_rays)
    c_z, rays_z, seen_rays_z = _recompute_reduced_rays(rays_y, sb, c_y, config)
    validated_facets = Set{HashKey}()
    oracle = SeparationOracle(A_sparse, Jvec; adapter=adapter, c=c_y, config=config)

    _algorithm_log(
        config,
        resumed ?
            "Resumed projected_cone_oracle from $(config.checkpoint_file) with $(length(rays_y)) rays, iter=$iter, lp_calls=$lp_calls" :
            "Initialized projected_cone_oracle with $(length(rays_y)) seed rays in working dimension $(size(sb.B, 2))/$d",
    )

    while iter < config.max_iterations
        iter += 1
        _algorithm_log(
            config,
            "Iteration $iter: rays=$(length(rays_z)), working_dim=$(size(sb.B, 2)), validated_facets=$(length(validated_facets)), lp_calls=$lp_calls",
        )

        G = _rationalized_ray_matrix(rays_z, config)
        facets_z = enumerate_facets(G, config; backend=backend)
        size(facets_z, 1) == 0 && return _result_from_current_cone(rays_z, sb, iter, lp_calls, config, backend, true)

        facet_order = collect(1:size(facets_z, 1))
        config.shuffle_facets && Random.shuffle!(facet_order)
        violation_found = false

        _algorithm_log(config, "  Enumerated $(size(facets_z, 1)) candidate facets")

        for facet_idx in facet_order
            g_z = Float64.(collect(facets_z[facet_idx, :]))
            g_key = hash_key_normal(g_z, config)
            g_key in validated_facets && continue

            h_y = lift(sb, g_z)
            obj, y_candidate, facet_lp_calls = solve_verified!(
                oracle,
                h_y;
                canonicalize=true,
                verify_tol=-config.violation_tol,
            )
            lp_calls += facet_lp_calls

            if obj < -config.violation_tol
                violation_found = true
                y_new = canonical_ray(y_candidate, config)
                expanded = maybe_expand!(sb, y_new, config)
                expanded && empty!(validated_facets)

                if expanded
                    c_z, rays_z, seen_rays_z = _recompute_reduced_rays(rays_y, sb, c_y, config)
                    _algorithm_log(config, "  Expanded working subspace to dimension $(size(sb.B, 2))/$d")
                end

                z_new = cy_normalize(project(sb, y_new), c_z, config)
                z_key = hash_key_ray(z_new, config)
                if z_key in seen_rays_z
                    push!(validated_facets, g_key)
                    _algorithm_log(config, "  Facet objective $(round(obj; digits=8)) produced an already-known ray; caching facet as validated")
                else
                    push!(rays_y, y_new)
                    if expanded
                        c_z, rays_z, seen_rays_z = _recompute_reduced_rays(rays_y, sb, c_y, config)
                    else
                        push!(rays_z, z_new)
                        push!(seen_rays_z, z_key)
                    end

                    if config.checkpoint_file !== nothing
                        B_to_save = size(sb.B, 2) == d ? nothing : sb.B
                        save_checkpoint(config.checkpoint_file, rays_y, iter, lp_calls, B_to_save, config)
                    end

                    _algorithm_log(
                        config,
                        "  Violation found: obj=$(round(obj; digits=8)); added ray $(canonical_ray(y_new, config))",
                    )
                end
                break
            else
                push!(validated_facets, g_key)
            end
        end

        if !violation_found
            _algorithm_log(config, "Converged after $iter iterations and $lp_calls LP calls")
            return _result_from_current_cone(rays_z, sb, iter, lp_calls, config, backend, true)
        end
    end

    _algorithm_log(config, "Reached max_iterations=$(config.max_iterations) after $iter iterations; returning current inner approximation")
    return _result_from_current_cone(rays_z, sb, iter, lp_calls, config, backend, false)
end

function projected_cone_oracle(
    A::AbstractMatrix{<:Real},
    J::AbstractVector{<:Integer};
    config::OracleConfig=OracleConfig(),
    solver::AbstractSolverAdapter=default_solver_adapter(),
    backend::AbstractNormalizBackend=NormalizCLIBackend(),
)
    return projected_cone_oracle(A, J, config, solver; backend=backend)
end

"""
    projected_cone(A, J; config, solver, adapter, backend)

Convenience wrapper for [`projected_cone_oracle`](@ref) with keyword-only arguments.

Equivalent to `projected_cone_oracle(A, J, config, solver; backend=backend)`.
Accepts both `solver` and `adapter` as keyword names (they are synonyms).
"""
function projected_cone(
    A::AbstractMatrix{<:Real},
    J::AbstractVector{<:Integer};
    config::OracleConfig=OracleConfig(),
    solver::AbstractSolverAdapter=default_solver_adapter(),
    adapter::AbstractSolverAdapter=solver,
    backend::AbstractNormalizBackend=NormalizCLIBackend(),
)
    return projected_cone_oracle(A, J, config, adapter; backend=backend)
end
