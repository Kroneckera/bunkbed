using Serialization
using SparseArrays
using LinearAlgebra
using JuMP
using HiGHS

export PAPER_FAMILY_F_PATH, PAPER_FAMILY_F_KEY, APPENDIX_ORDER_IDS
export load_feasible_tuples_raw, load_feasible_tuples, paper_family_feasible_tuples
export baseline_to_appendix_id, appendix_to_baseline_id, reorder_pair_matrix
export archive_tuple_to_paper_pairs, phi_index, inverse_phi_index, basis_direction
export build_constraint_matrix, find_inequality, recover_target_certificate, extract_quadratic_polynomial, verify_certificate, systematic_inequality_search
export appendixA_certificate_11, appendixA_certificate_12, inequality7_proof_potentials
export canonicalize_integer_ray, canonicalize_phi_tables, polynomial_signature

const MOI = JuMP.MOI

const PAPER_FAMILY_F_PATH = normpath(joinpath(@__DIR__, "..", "data", "valid_partition_tuples_nobs3_notebook.jls"))
const PAPER_FAMILY_F_KEY = "tuples_ids_prefix"
const APPENDIX_ORDER_IDS = PAPER_J3_ORDER_IDS

function load_feasible_tuples_raw(path::AbstractString=PAPER_FAMILY_F_PATH; key::AbstractString=PAPER_FAMILY_F_KEY)
    artifact = deserialize(path)
    return artifact isa Dict ? artifact[key] : artifact
end

function load_feasible_tuples(
    path::AbstractString=PAPER_FAMILY_F_PATH;
    key::AbstractString=PAPER_FAMILY_F_KEY,
    n_obs::Int=3,
    id_base::Symbol=:auto,
)
    n_obs == 3 || throw(ArgumentError("the bundled feasible-tuple archive currently supports only n_obs=3"))
    raw_tuples = load_feasible_tuples_raw(path; key=key)
    return [normalize_archive_feasible_tuple(tuple_like, n_obs; id_base=id_base) for tuple_like in raw_tuples]
end

function paper_family_feasible_tuples(; id_base::Symbol=:auto)
    return load_feasible_tuples(; id_base=id_base)
end

function baseline_to_appendix_id(id::Integer)
    pos = findfirst(==(PartitionID(id)), APPENDIX_ORDER_IDS)
    pos === nothing && throw(ArgumentError("invalid baseline partition id: $id"))
    return PartitionID(pos - 1)
end

function appendix_to_baseline_id(id::Integer)
    idx = Int(id) + 1
    1 <= idx <= length(APPENDIX_ORDER_IDS) || throw(BoundsError(APPENDIX_ORDER_IDS, idx))
    return APPENDIX_ORDER_IDS[idx]
end

function reorder_pair_matrix(matrix::AbstractMatrix, n_obs::Int; from_order::Symbol=:baseline, to_order::Symbol=:baseline)
    return reorder_partition_matrix(matrix, n_obs; from_order=from_order, to_order=to_order)
end

"""
    archive_tuple_to_paper_pairs(tuple_like, n_obs, m; id_base=:auto)

Compatibility wrapper for the legacy paper-family archive.
For the bundled 8-tuple artifact, the incoming tree order is `(T0,T2,T1,T3)`;
this helper returns the normalized paper order `(T0,T1,T2,T3)`.
"""
function archive_tuple_to_paper_pairs(tuple_like, n_obs::Int, m::Int; id_base::Symbol=:auto)
    if n_obs == 3 && m == 4
        return normalize_archive_feasible_tuple(tuple_like, n_obs; id_base=id_base)
    end
    return normalize_feasible_tuple(tuple_like, n_obs, m; id_base=id_base)
end

function phi_index(k::Integer, p::Integer, pbar::Integer, n_obs::Int)
    bell = bell_number(n_obs)
    1 <= k || throw(ArgumentError("tree index must be positive, got $k"))
    0 <= Int(p) < bell || throw(ArgumentError("partition id p=$p outside valid range 0:$(bell - 1)"))
    0 <= Int(pbar) < bell || throw(ArgumentError("partition id pbar=$pbar outside valid range 0:$(bell - 1)"))
    return (Int(k) - 1) * bell * bell + Int(p) * bell + Int(pbar) + 1
end

function inverse_phi_index(index::Integer, n_obs::Int)
    bell = bell_number(n_obs)
    index >= 1 || throw(ArgumentError("column index must be positive, got $index"))
    block, local_idx = divrem(Int(index) - 1, bell * bell)
    p, pbar = divrem(local_idx, bell)
    return (tree=block + 1, p=PartitionID(p), pbar=PartitionID(pbar))
end

function basis_direction(nvars::Int, index::Integer)
    1 <= abs(Int(index)) <= nvars || throw(BoundsError(1:nvars, index))
    direction = zeros(Float64, nvars)
    direction[abs(Int(index))] = Int(index) < 0 ? -1.0 : 1.0
    return direction
end

function _coerce_phi_tables(phi_tables, n_obs::Int, m::Int)
    bell = bell_number(n_obs)
    if phi_tables isa AbstractVector && !isempty(phi_tables) && first(phi_tables) isa AbstractMatrix
        length(phi_tables) == m || throw(ArgumentError("expected $m potential tables, got $(length(phi_tables))"))
        return [Matrix(table) for table in phi_tables]
    elseif phi_tables isa AbstractArray && ndims(phi_tables) == 3
        if size(phi_tables) == (m, bell, bell)
            return [Matrix(phi_tables[k, :, :]) for k in 1:m]
        elseif size(phi_tables) == (bell, bell, m)
            return [Matrix(phi_tables[:, :, k]) for k in 1:m]
        else
            throw(ArgumentError("unsupported 3D potential-table shape $(size(phi_tables)); expected ($m,$bell,$bell) or ($bell,$bell,$m)"))
        end
    elseif phi_tables isa AbstractVector
        length(phi_tables) == m * bell * bell || throw(ArgumentError("expected flattened potential vector of length $(m * bell * bell), got $(length(phi_tables))"))
        tables = Vector{Matrix{eltype(phi_tables)}}(undef, m)
        for k in 1:m
            table = Matrix{eltype(phi_tables)}(undef, bell, bell)
            for p in all_partition_ids(n_obs), pbar in all_partition_ids(n_obs)
                table[Int(p) + 1, Int(pbar) + 1] = phi_tables[phi_index(k, p, pbar, n_obs)]
            end
            tables[k] = table
        end
        return tables
    else
        throw(ArgumentError("unsupported potential-table container type: $(typeof(phi_tables))"))
    end
end

function _infer_m(phi_tables, n_obs::Int)
    bell = bell_number(n_obs)
    if phi_tables isa AbstractVector && !isempty(phi_tables) && first(phi_tables) isa AbstractMatrix
        return length(phi_tables)
    elseif phi_tables isa AbstractArray && ndims(phi_tables) == 3
        if size(phi_tables, 1) == bell && size(phi_tables, 2) == bell
            return size(phi_tables, 3)
        elseif size(phi_tables, 2) == bell && size(phi_tables, 3) == bell
            return size(phi_tables, 1)
        end
    elseif phi_tables isa AbstractVector
        return div(length(phi_tables), bell * bell)
    end
    throw(ArgumentError("cannot infer number of trees from potential-table container $(typeof(phi_tables))"))
end

function _flatten_phi_tables(phi_tables, n_obs::Int, m::Int)
    tables = _coerce_phi_tables(phi_tables, n_obs, m)
    bell = bell_number(n_obs)
    values = Vector{eltype(tables[1])}(undef, m * bell * bell)
    for k in 1:m, p in all_partition_ids(n_obs), pbar in all_partition_ids(n_obs)
        values[phi_index(k, p, pbar, n_obs)] = tables[k][Int(p) + 1, Int(pbar) + 1]
    end
    return values
end

function build_constraint_matrix(feasible_tuples, n_obs::Int, m::Int; id_base::Symbol=:auto)
    rows = length(feasible_tuples)
    bell = bell_number(n_obs)
    cols = m * bell * bell
    resolved_base = id_base === :auto ? _detect_partition_id_base(feasible_tuples, bell) : id_base
    I = Vector{Int}(undef, rows * m)
    J = Vector{Int}(undef, rows * m)
    V = fill(Int8(1), rows * m)

    offset = 1
    for (row, tuple_like) in enumerate(feasible_tuples)
        pairs = normalize_feasible_tuple(tuple_like, n_obs, m; id_base=resolved_base)
        for k in 1:m
            p, pbar = pairs[k]
            I[offset] = row
            J[offset] = phi_index(k, p, pbar, n_obs)
            offset += 1
        end
    end

    return sparse(I, J, V, rows, cols)
end

function verify_certificate(
    phi_tables,
    feasible_tuples;
    n_obs::Int=3,
    m::Int=4,
    id_base::Symbol=:auto,
    max_witnesses::Int=3,
    phi_order::Symbol=default_partition_order(n_obs),
)
    tables = _coerce_phi_tables(phi_tables, n_obs, m)
    if phi_order !== :baseline
        tables = [reorder_pair_matrix(table, n_obs; from_order=phi_order, to_order=:baseline) for table in tables]
    end
    bell = bell_number(n_obs)
    resolved_base = id_base === :auto ? _detect_partition_id_base(feasible_tuples, bell) : id_base
    min_value = nothing
    witness_indices = Int[]
    witness_tuples = Any[]
    negatives = 0

    for (idx, tuple_like) in enumerate(feasible_tuples)
        pairs = normalize_feasible_tuple(tuple_like, n_obs, m; id_base=resolved_base)
        total = zero(tables[1][1, 1])
        for k in 1:m
            p, pbar = pairs[k]
            total += tables[k][Int(p) + 1, Int(pbar) + 1]
        end
        if min_value === nothing || total < min_value
            min_value = total
            empty!(witness_indices)
            empty!(witness_tuples)
            push!(witness_indices, idx)
            push!(witness_tuples, tuple_like)
        elseif total == min_value && length(witness_indices) < max_witnesses
            push!(witness_indices, idx)
            push!(witness_tuples, tuple_like)
        end
        negatives += total < zero(total)
    end

    min_value === nothing && (min_value = zero(Int))
    return (
        feasible = negatives == 0,
        minimum = min_value,
        negatives = negatives,
        witness_indices = witness_indices,
        witness_tuples = witness_tuples,
    )
end

function extract_quadratic_polynomial(
    phi_tables,
    n_obs::Int;
    phi_order::Symbol=default_partition_order(n_obs),
    output_order::Symbol=default_partition_order(n_obs),
)
    tables = _coerce_phi_tables(phi_tables, n_obs, _infer_m(phi_tables, n_obs))
    if phi_order !== :baseline
        tables = [reorder_pair_matrix(table, n_obs; from_order=phi_order, to_order=:baseline) for table in tables]
    end
    bell = bell_number(n_obs)
    aggregated = zeros(promote_type(map(eltype, tables)...), bell, bell)
    for table in tables
        aggregated .+= table
    end

    ids = partition_order_ids(n_obs; order=output_order)
    labels = [partition_label(n_obs, pid) for pid in ids]
    diagonal = Dict{PartitionID, eltype(aggregated)}()
    offdiag = Dict{Tuple{PartitionID, PartitionID}, eltype(aggregated)}()

    for pid in ids
        diagonal[pid] = aggregated[Int(pid) + 1, Int(pid) + 1]
    end
    for i in eachindex(ids), j in i+1:length(ids)
        pi = ids[i]
        pj = ids[j]
        offdiag[(pi, pj)] = aggregated[Int(pi) + 1, Int(pj) + 1] + aggregated[Int(pj) + 1, Int(pi) + 1]
    end

    return (
        order = ids,
        labels = labels,
        aggregated = aggregated,
        diagonal = diagonal,
        offdiag = offdiag,
    )
end

function canonicalize_integer_ray(values; tol::Real=1e-8, normalize_sign::Bool=false)
    rationals = Rational{BigInt}[]
    for value in values
        x = value isa Rational ? Rational{BigInt}(value) : rationalize(BigInt, float(value); tol=tol)
        if abs(float(x)) <= tol
            x = 0//1
        end
        push!(rationals, x)
    end
    if all(iszero, rationals)
        return fill(BigInt(0), length(rationals))
    end
    lcm_den = foldl(lcm, (denominator(x) for x in rationals if !iszero(x)); init=BigInt(1))
    integers = [numerator(x * lcm_den) for x in rationals]
    gcd_num = foldl(gcd, (abs(x) for x in integers if !iszero(x)); init=BigInt(0))
    gcd_num != 0 && (integers = [div(x, gcd_num) for x in integers])
    if normalize_sign
        first_nonzero = findfirst(!iszero, integers)
        first_nonzero !== nothing && integers[first_nonzero] < 0 && (integers = [-x for x in integers])
    end
    return integers
end

function canonicalize_phi_tables(phi_tables, n_obs::Int, m::Int; tol::Real=1e-8)
    primitive = canonicalize_integer_ray(_flatten_phi_tables(phi_tables, n_obs, m); tol=tol)
    return _coerce_phi_tables(primitive, n_obs, m)
end

function _symmetrized_coefficient_vector(poly)
    ids = poly.order
    values = BigInt[]
    for pid in ids
        push!(values, BigInt(poly.diagonal[pid]))
    end
    for i in eachindex(ids), j in i+1:length(ids)
        push!(values, BigInt(poly.offdiag[(ids[i], ids[j])]))
    end
    return values
end

function polynomial_signature(
    phi_tables,
    n_obs::Int;
    tol::Real=1e-8,
    phi_order::Symbol=default_partition_order(n_obs),
    output_order::Symbol=default_partition_order(n_obs),
)
    poly = extract_quadratic_polynomial(phi_tables, n_obs; phi_order=phi_order, output_order=output_order)
    return canonicalize_integer_ray(_symmetrized_coefficient_vector(poly); tol=tol, normalize_sign=false)
end

function _dense_linear_form(spec, nvars::Int)
    if spec isa AbstractVector
        length(spec) == nvars || throw(DimensionMismatch("expected linear form of length $nvars, got length $(length(spec))"))
        return Float64.(spec)
    elseif spec isa Integer
        return basis_direction(nvars, spec)
    else
        throw(ArgumentError("unsupported linear-form specification: $(typeof(spec))"))
    end
end

function find_inequality(M_F::SparseMatrixCSC, objective, normalization; n_obs::Union{Nothing,Int}=nothing, m::Union{Nothing,Int}=nothing, feasible_tuples=nothing, rationalize_tol::Real=1e-8, time_limit_sec::Union{Nothing,Real}=nothing)
    nvars = size(M_F, 2)
    c = _dense_linear_form(objective, nvars)
    d = _dense_linear_form(normalization, nvars)

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    if time_limit_sec !== nothing
        set_optimizer_attribute(model, "time_limit", Float64(time_limit_sec))
    end

    @variable(model, phi[1:nvars])
    @constraint(model, M_F * phi .>= 0)
    @constraint(model, dot(d, phi) == 1)
    @objective(model, Min, dot(c, phi))
    optimize!(model)

    status = termination_status(model)
    pstatus = primal_status(model)
    has_vals = has_values(model)
    phi_value = has_vals ? value.(phi) : nothing
    result = (
        termination_status = status,
        primal_status = pstatus,
        objective_value = has_vals ? objective_value(model) : nothing,
        phi = phi_value,
        objective = c,
        normalization = d,
        nvars = nvars,
    )

    if !has_vals || n_obs === nothing || m === nothing
        return result
    end

    raw_tables = _coerce_phi_tables(phi_value, n_obs, m)
    primitive_tables = canonicalize_phi_tables(phi_value, n_obs, m; tol=rationalize_tol)
    primitive_vector = _flatten_phi_tables(primitive_tables, n_obs, m)
    primitive_poly = extract_quadratic_polynomial(primitive_tables, n_obs; phi_order=:baseline)
    primitive_signature = polynomial_signature(primitive_tables, n_obs; tol=rationalize_tol, phi_order=:baseline)
    verification = feasible_tuples === nothing ? nothing : verify_certificate(primitive_tables, feasible_tuples; n_obs=n_obs, m=m, phi_order=:baseline)

    return merge(result, (
        raw_tables = raw_tables,
        primitive_tables = primitive_tables,
        primitive_phi = primitive_vector,
        polynomial = primitive_poly,
        polynomial_signature = primitive_signature,
        verification = verification,
    ))
end


function recover_target_certificate(
    M_F::SparseMatrixCSC,
    target_phi,
    normalization;
    n_obs::Int,
    m::Int,
    feasible_tuples=nothing,
    rationalize_tol::Real=1e-8,
    time_limit_sec::Union{Nothing,Real}=nothing,
    phi_order::Symbol=default_partition_order(n_obs),
)
    nvars = size(M_F, 2)
    target_tables = _coerce_phi_tables(target_phi, n_obs, m)
    if phi_order !== :baseline
        target_tables = [reorder_pair_matrix(table, n_obs; from_order=phi_order, to_order=:baseline) for table in target_tables]
    end
    target = Float64.(_flatten_phi_tables(target_tables, n_obs, m))
    d = _dense_linear_form(normalization, nvars)
    scale = dot(d, target)
    abs(scale) > 1e-12 || throw(ArgumentError("normalization is orthogonal to the target certificate"))
    target_scaled = target ./ scale

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    if time_limit_sec !== nothing
        set_optimizer_attribute(model, "time_limit", Float64(time_limit_sec))
    end

    @variable(model, phi[1:nvars])
    @variable(model, slack[1:nvars] >= 0)
    @constraint(model, M_F * phi .>= 0)
    @constraint(model, dot(d, phi) == 1)
    @constraint(model, phi .- target_scaled .<= slack)
    @constraint(model, target_scaled .- phi .<= slack)
    @objective(model, Min, sum(slack))
    optimize!(model)

    status = termination_status(model)
    pstatus = primal_status(model)
    has_vals = has_values(model)
    phi_value = has_vals ? value.(phi) : nothing
    result = (
        termination_status = status,
        primal_status = pstatus,
        objective_value = has_vals ? objective_value(model) : nothing,
        phi = phi_value,
        normalization = d,
        target_scaled = target_scaled,
    )

    if !has_vals
        return result
    end

    raw_tables = _coerce_phi_tables(phi_value, n_obs, m)
    primitive_tables = canonicalize_phi_tables(phi_value, n_obs, m; tol=rationalize_tol)
    primitive_vector = _flatten_phi_tables(primitive_tables, n_obs, m)
    primitive_poly = extract_quadratic_polynomial(primitive_tables, n_obs; phi_order=:baseline)
    primitive_signature = polynomial_signature(primitive_tables, n_obs; tol=rationalize_tol, phi_order=:baseline)
    verification = feasible_tuples === nothing ? nothing : verify_certificate(primitive_tables, feasible_tuples; n_obs=n_obs, m=m, phi_order=:baseline)
    target_signature = polynomial_signature(target_tables, n_obs; tol=rationalize_tol, phi_order=:baseline)

    return merge(result, (
        raw_tables = raw_tables,
        primitive_tables = primitive_tables,
        primitive_phi = primitive_vector,
        polynomial = primitive_poly,
        polynomial_signature = primitive_signature,
        verification = verification,
        target_signature = target_signature,
    ))
end

function systematic_inequality_search(M_F::SparseMatrixCSC, n_obs::Int, m::Int; feasible_tuples=nothing, anchors=nothing, directions=nothing, rationalize_tol::Real=1e-8, time_limit_sec::Real=5.0, total_time_limit_sec::Union{Nothing,Real}=nothing)
    nvars = size(M_F, 2)
    anchors === nothing && (anchors = collect(1:nvars))
    directions === nothing && (directions = vcat(collect(1:nvars), collect(-1:-1:-nvars)))

    catalog = NamedTuple[]
    seen = Dict{Tuple{Vararg{BigInt}}, Int}()
    solve_times = Float64[]
    started = time()
    attempted = 0

    for anchor in anchors
        normalization = basis_direction(nvars, anchor)
        for direction in directions
            if total_time_limit_sec !== nothing && (time() - started) > total_time_limit_sec
                return (
                    catalog = catalog,
                    attempted = attempted,
                    unique = length(catalog),
                    elapsed_seconds = time() - started,
                    solve_times = solve_times,
                    anchors = collect(anchors),
                    directions = collect(directions),
                    timed_out = true,
                )
            end

            attempted += 1
            solve_started = time()
            result = find_inequality(
                M_F,
                basis_direction(nvars, direction),
                normalization;
                n_obs=n_obs,
                m=m,
                feasible_tuples=feasible_tuples,
                rationalize_tol=rationalize_tol,
                time_limit_sec=time_limit_sec,
            )
            solve_elapsed = time() - solve_started
            push!(solve_times, solve_elapsed)

            if !hasproperty(result, :polynomial_signature) || result.termination_status != MOI.OPTIMAL
                continue
            end

            signature = Tuple(result.polynomial_signature)
            if !haskey(seen, signature)
                push!(catalog, merge(result, (
                    anchor = anchor,
                    direction = direction,
                    solve_time_sec = solve_elapsed,
                )))
                seen[signature] = length(catalog)
            end
        end
    end

    return (
        catalog = catalog,
        attempted = attempted,
        unique = length(catalog),
        elapsed_seconds = time() - started,
        solve_times = solve_times,
        anchors = collect(anchors),
        directions = collect(directions),
        timed_out = false,
    )
end

function _mat(rows::Vector{Vector{Int}})
    length(rows) == 5 || throw(ArgumentError("expected 5 rows"))
    all(length(row) == 5 for row in rows) || throw(ArgumentError("expected 5 columns"))
    return reduce(vcat, (reshape(row, 1, :) for row in rows))
end

function appendixA_certificate_11(; order::Symbol=default_partition_order(3))
    raw_blocks = [
        zeros(Int, 5, 5),
        _mat([
            [-1,  0,  0, -1,  0],
            [ 0,  0,  0,  0,  0],
            [ 1,  0,  0,  1,  0],
            [ 1,  1,  1,  1,  1],
            [ 0,  0,  0,  0,  0],
        ]),
        _mat([
            [ 0, -1, -1,  0, -1],
            [ 0,  0,  0,  0,  0],
            [ 0,  1,  1,  0,  1],
            [ 0,  0,  0,  0,  0],
            [ 1,  1,  1,  1,  1],
        ]),
        _mat([
            [ 1,  1,  1,  1,  1],
            [-1,  0,  0, -1, -1],
            [-1,  0,  0, -1, -1],
            [ 0,  0,  0, -1,  0],
            [ 0,  0,  0,  0, -1],
        ]),
    ]
    return [reorder_pair_matrix(block, 3; from_order=:paper, to_order=order) for block in raw_blocks]
end

function appendixA_certificate_12(; order::Symbol=default_partition_order(3))
    raw_blocks = [
        zeros(Int, 5, 5),
        _mat([
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]),
        zeros(Int, 5, 5),
        _mat([
            [ 0,  0,  0,  0,  0],
            [ 0, -1,  0,  0, -1],
            [ 0, -1,  0,  0, -1],
            [ 0, -1, -1,  0, -1],
            [ 0,  0,  0,  0,  0],
        ]),
    ]
    return [reorder_pair_matrix(block, 3; from_order=:paper, to_order=order) for block in raw_blocks]
end

"""
    inequality7_proof_potentials(; order=default_partition_order(3), swap_T3_pair=false)

Return the 4 potential tables (one per tree) appearing in the proof of inequality (7) (Prop. 10.1 / `sections/proofs/ineq7.tex`).

Important convention note: the proof's 4th tree corresponds (at condensation level) to the **complement** of the paper-family `T3`,
so when verifying against the archived paper-family feasible set `F ⊂ (J₃²)⁴` one must swap the 4th pair `(p3,p̄3)`. Passing
`swap_T3_pair=true` implements this by transposing only the 4th table (which preserves the induced symmetric quadratic inequality).
"""
function inequality7_proof_potentials(; order::Symbol=default_partition_order(3), swap_T3_pair::Bool=false)
    order in (:baseline, :appendixA, :paper, :paper_n3) || throw(ArgumentError("unsupported order: $order"))
    tables = [zeros(Int, 5, 5) for _ in 1:4]

    # Baseline J3 ids: 123=0, 12|3=1, 13|2=2, 1|23=3, 1|2|3=4.
    tables[1][4, :] .= 1   # φ0(p, p̄) = 1{p = 1|23}
    tables[2][2, :] .= 1   # φ1(p, p̄) = 1{p = 12|3}
    tables[3][3, :] .= 1   # φ2(p, p̄) = 1{p = 13|2}
    for p in (1, 2, 3), pbar in (4, 5)
        tables[4][p, pbar] = -1
    end

    if swap_T3_pair
        tables[4] = permutedims(tables[4], (2, 1))
    end

    return order === :baseline ? tables : [reorder_pair_matrix(table, 3; from_order=:baseline, to_order=order) for table in tables]
end
