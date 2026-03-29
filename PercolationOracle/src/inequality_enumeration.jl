using SparseArrays
using ProjectedConeOracle

export InequalityEnumerationResult, enumerate_all_inequalities

"""
    InequalityEnumerationResult

Result of the D1→D2 inequality-enumeration pipeline for the archived paper family.

# Fields
- `facet_normals::Vector{Vector{Float64}}`: projected-cone facet normals / meta-constraints in symmetric-coordinate space. These are **not** the final percolation inequalities.
- `rays::Vector{Vector{Float64}}`: extreme rays of the projected cone; these are the valid percolation inequalities.
- `labels::Vector{String}`: symbolic labels for the symmetric coordinates.
- `formatted_inequalities::Vector{String}`: human-readable strings for the ray inequalities.
- `projection_result::ProjectedConeResult`: raw result returned by `ProjectedConeOracle`.
- `timing::Float64`: wall-clock seconds spent in the projected-cone stage.
"""
struct InequalityEnumerationResult
    facet_normals::Vector{Vector{Float64}}
    rays::Vector{Vector{Float64}}
    labels::Vector{String}
    formatted_inequalities::Vector{String}
    projection_result::ProjectedConeResult
    timing::Float64
end

function Base.getproperty(result::InequalityEnumerationResult, name::Symbol)
    if name === :inequalities
        return getfield(result, :facet_normals)
    end
    return getfield(result, name)
end

function Base.propertynames(::InequalityEnumerationResult, private::Bool=false)
    names = (:facet_normals, :rays, :labels, :formatted_inequalities, :projection_result, :timing)
    return private ? (names..., :inequalities) : (names..., :inequalities)
end

function _require_paper_family_scope(n_obs::Int, m::Int)
    n_obs == 3 || throw(ArgumentError("enumerate_all_inequalities currently supports only n_obs=3 via the archived paper-family feasible tuples; got n_obs=$n_obs"))
    1 <= m <= 4 || throw(ArgumentError("enumerate_all_inequalities currently supports 1 <= m <= 4 for the archived paper family; got m=$m"))
    return nothing
end

function _project_archived_tuples(n_obs::Int, m::Int)
    _require_paper_family_scope(n_obs, m)
    raw_tuples = load_feasible_tuples()
    projected = Set{Any}()
    for tuple_like in raw_tuples
        pairs = archive_tuple_to_paper_pairs(tuple_like, n_obs, 4)
        push!(projected, Tuple(pairs[i] for i in 1:m))
    end
    return collect(projected)
end

function _mu_label(n_obs::Int, pid::Integer)
    return "mu($(partition_label(n_obs, pid)))"
end

function _monomial_label(n_obs::Int, p::Integer, q::Integer)
    lp = _mu_label(n_obs, p)
    lq = _mu_label(n_obs, q)
    return Int(p) == Int(q) ? "$(lp)^2" : "$(lp)*$(lq)"
end

function _symmetric_coordinates(n_obs::Int)
    ids = collect(all_partition_ids(n_obs))
    coordinates = Tuple{PartitionID, PartitionID}[]
    labels = String[]

    for pid in ids
        push!(coordinates, (pid, pid))
        push!(labels, _monomial_label(n_obs, pid, pid))
    end
    for i in eachindex(ids), j in (i + 1):length(ids)
        pi = ids[i]
        pj = ids[j]
        push!(coordinates, (pi, pj))
        push!(labels, _monomial_label(n_obs, pi, pj))
    end

    return coordinates, labels
end

function _append_signed_equality_rows!(I::Vector{Int}, J::Vector{Int}, V::Vector{Float64}, row_idx::Base.RefValue{Int}, coeffs::Vector{Tuple{Int, Float64}})
    row_idx[] += 1
    for (col, val) in coeffs
        push!(I, row_idx[])
        push!(J, col)
        push!(V, val)
    end

    row_idx[] += 1
    for (col, val) in coeffs
        push!(I, row_idx[])
        push!(J, col)
        push!(V, -val)
    end
    return nothing
end

function _extend_with_symmetric_variables(M_F::SparseMatrixCSC, n_obs::Int, m::Int)
    coordinates, labels = _symmetric_coordinates(n_obs)
    num_rows, num_phi = size(M_F)
    num_sym = length(coordinates)

    I0, J0, V0 = findnz(M_F)
    I = Int.(I0)
    J = Int.(J0)
    V = Float64.(V0)
    row_idx = Ref(num_rows)
    projection_indices = Int[]

    for (coord_idx, (p, q)) in enumerate(coordinates)
        sym_col = num_phi + coord_idx
        push!(projection_indices, sym_col)
        coeffs = Tuple{Int, Float64}[(sym_col, 1.0)]
        if p == q
            for k in 1:m
                push!(coeffs, (phi_index(k, p, q, n_obs), -1.0))
            end
        else
            for k in 1:m
                push!(coeffs, (phi_index(k, p, q, n_obs), -1.0))
                push!(coeffs, (phi_index(k, q, p, n_obs), -1.0))
            end
        end
        _append_signed_equality_rows!(I, J, V, row_idx, coeffs)
    end

    A = sparse(I, J, V, row_idx[], num_phi + num_sym)
    return (A=A, projection_indices=projection_indices, labels=labels, coordinates=coordinates)
end

function _canonical_signature(values::AbstractVector{<:Real}; tol::Real=1e-8)
    return canonicalize_integer_ray(values; tol=tol, normalize_sign=true)
end

function _format_coefficient(abs_coeff::Integer, label::String)
    return abs_coeff == 1 ? label : string(abs_coeff, "*", label)
end

function _format_inequality(values::AbstractVector{<:Real}, labels::AbstractVector{<:AbstractString}; tol::Real=1e-8)
    signature = _canonical_signature(values; tol=tol)
    terms = String[]
    for (coeff, label) in zip(signature, labels)
        coeff == 0 && continue
        body = _format_coefficient(abs(coeff), String(label))
        if isempty(terms)
            push!(terms, coeff < 0 ? "-$(body)" : body)
        else
            push!(terms, coeff < 0 ? "- $(body)" : "+ $(body)")
        end
    end
    return isempty(terms) ? "0 >= 0" : join(terms, " ") * " >= 0"
end

function _formatted_inequalities(rays::Vector{Vector{Float64}}, labels::Vector{String})
    return [_format_inequality(ray, labels) for ray in rays]
end

"""
    enumerate_all_inequalities(n_obs, m; solver=HiGHSAdapter(), verbose=true, max_iterations=5000)

Run the production inequality-enumeration pipeline that connects the archived D1 paper-family
feasible tuples to the D2 projected-cone oracle.

Current scope: the Stage 1 tuple source is the archived paper-family dataset loaded by
`load_feasible_tuples()`, so the public entry point currently supports only `n_obs=3` and
`1 <= m <= 4`, where `m` counts the prefix length in the normalized paper order `(T0,T1,T2,T3)`.

The returned `rays` are the actual valid inequalities in symmetric coordinate space;
`facet_normals` contains the projected-cone meta-constraints for the same coordinate system.
A backward-compatible alias `result.inequalities` is kept for local scripts/notebooks.
"""
function enumerate_all_inequalities(
    n_obs::Int,
    m::Int;
    solver::AbstractSolverAdapter=HiGHSAdapter(),
    verbose::Bool=true,
    max_iterations::Int=5000,
)
    _require_paper_family_scope(n_obs, m)
    projected_tuples = _project_archived_tuples(n_obs, m)
    M_F = build_constraint_matrix(projected_tuples, n_obs, m; id_base=:zero_based)
    extension = _extend_with_symmetric_variables(M_F, n_obs, m)
    config = OracleConfig(verbose=verbose, max_iterations=max_iterations)

    t0 = time()
    projection_result = projected_cone_oracle(
        extension.A,
        extension.projection_indices,
        ;
        config=config,
        solver=solver,
    )
    timing = time() - t0
    formatted = _formatted_inequalities(projection_result.rays, extension.labels)
    return InequalityEnumerationResult(
        projection_result.facets,
        projection_result.rays,
        extension.labels,
        formatted,
        projection_result,
        timing,
    )
end

function enumerate_all_inequalities(
    n_obs::Int,
    m::Int,
    solver::AbstractSolverAdapter;
    verbose::Bool=true,
    max_iterations::Int=5000,
)
    return enumerate_all_inequalities(n_obs, m; solver=solver, verbose=verbose, max_iterations=max_iterations)
end
