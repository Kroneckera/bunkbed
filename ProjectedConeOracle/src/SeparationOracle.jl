"""
    SeparationOracle(A, J; adapter, c, config)

LP-based separation oracle for the projected cone `D = π_J({x : Ax ≥ 0})`.

Given a direction `h ∈ ℝᵈ`, solves `min h'y` subject to `y ∈ D`, `c'y = 1`.
If the minimum is negative, the minimizer is a new extreme ray witness.

# Construction
- `A::SparseMatrixCSC{Float64,Int}` — cone constraint matrix (m × n).
- `J::Vector{Int}` — projection indices.
- `c::AbstractVector{<:Real}` — normalization vector in `relint(D*)`.
- `adapter::AbstractSolverAdapter` — LP solver backend.

# Usage
```julia
obj, y = solve_oracle!(oracle, h)           # single-phase
obj, y, lps = solve_verified!(oracle, h)    # two-phase (barrier + simplex)
```
"""
struct SeparationOracle
    model::Model
    u::Vector{VariableRef}
    v::Vector{VariableRef}
    y::Vector{VariableRef}
    J::Vector{Int}
    d::Int
    config::OracleConfig
    adapter::AbstractSolverAdapter
    yval::Vector{Float64}
end

function SeparationOracle(
    A::SparseMatrixCSC{Float64,Int},
    J::Vector{Int};
    adapter::AbstractSolverAdapter=HiGHSAdapter(),
    c::AbstractVector{<:Real},
    config::OracleConfig=OracleConfig(),
)
    m, n = size(A)
    d = length(J)
    length(c) == d || error("SeparationOracle: length(c)=$(length(c)) must equal d=$d")

    model = create_model(adapter)

    @variable(model, u[1:n] >= 0)
    @variable(model, v[1:n] >= 0)
    @variable(model, y[1:d])
    @constraint(model, [t=1:d], y[t] == u[J[t]] - v[J[t]])

    rows = [Vector{Tuple{Int,Float64}}() for _ in 1:m]
    for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            i = A.rowval[ptr]
            push!(rows[i], (col, A.nzval[ptr]))
        end
    end

    for i in 1:m
        lst = rows[i]
        if !isempty(lst)
            @constraint(model, sum(val * (u[col] - v[col]) for (col, val) in lst) >= 0)
        end
    end

    cvec = Float64.(collect(c))
    @constraint(model, sum(cvec[t] * y[t] for t in 1:d) == 1.0)
    @objective(model, Min, sum(0.0 * y[t] for t in 1:d))

    return SeparationOracle(model, u, v, y, collect(J), d, config, adapter, zeros(Float64, d))
end

"""
    set_solver_method!(oracle::SeparationOracle, method::Int, crossover::Int=0) -> Bool

Switch the LP solver method on `oracle`'s model. If `method == 1`, applies
[`configure_verified!`](@ref); otherwise applies [`configure_fast!`](@ref).
Always returns `true`.
"""
function set_solver_method!(oracle::SeparationOracle, method::Int, crossover::Int=0)
    if method == 1
        configure_verified!(oracle.adapter, oracle.model)
    else
        configure_fast!(oracle.adapter, oracle.model)
    end
    return true
end

function _optimize_with_recovery!(oracle::SeparationOracle)
    optimize!(oracle.model)
    status = termination_status(oracle.model)
    if status == MOI.NUMERICAL_ERROR
        configure_recovery!(oracle.adapter, oracle.model)
        optimize!(oracle.model)
        status = termination_status(oracle.model)
        configure_default!(oracle.adapter, oracle.model)
    end
    return status
end

"""
    solve_oracle!(oracle::SeparationOracle, h; canonicalize=true) -> (obj, y)

Solve the separation LP `min h'y` subject to `y ∈ D`, `c'y = 1` using the oracle's current
solver configuration. Returns the objective value and the optimizer (canonicalized by default).

Errors on infeasible, unbounded, or non-optimal status. Automatically retries with
[`configure_recovery!`](@ref) on `NUMERICAL_ERROR`.

See also: [`solve_verified!`](@ref), [`SeparationOracle`](@ref).
"""
function solve_oracle!(oracle::SeparationOracle, h::AbstractVector{<:Real}; canonicalize::Bool=true)
    d = oracle.d
    length(h) == d || error("solve_oracle!: length(h)=$(length(h)) must equal d=$d")

    for t in 1:d
        set_objective_coefficient(oracle.model, oracle.y[t], Float64(h[t]))
    end

    status = _optimize_with_recovery!(oracle)
    if status == MOI.INFEASIBLE
        error("solve_oracle!: normalization slice is infeasible (projected cone may be {0})")
    elseif status in (MOI.DUAL_INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)
        error(
            "solve_oracle!: LP solve failed with status: $status " *
            "(projected cone may have lineality / non-pointed projected directions in the chosen normalization slice)"
        )
    elseif status != MOI.OPTIMAL
        error("LP solve failed with status: $status")
    end

    obj = objective_value(oracle.model)
    for t in 1:d
        oracle.yval[t] = value(oracle.y[t])
    end

    if canonicalize
        return obj, canonical_ray(oracle.yval, oracle.config)
    else
        return obj, copy(oracle.yval)
    end
end

"""
    solve_verified!(oracle, h; canonicalize=true, verify_tol=nothing) -> (obj, y, lp_count)

Two-phase LP solve: first [`configure_fast!`](@ref) (barrier), then [`configure_verified!`](@ref)
(simplex) for a numerically reliable basic solution. Skips the verification phase if the
objective exceeds `verify_tol` (non-violating direction).

Returns `(objective_value, ray, lp_call_count)` where `lp_call_count` is 1 or 2.

See also: [`solve_oracle!`](@ref), [`supports_two_phase`](@ref).
"""
function solve_verified!(
    oracle::SeparationOracle,
    h::AbstractVector{<:Real};
    canonicalize::Bool=true,
    verify_tol::Union{Float64,Nothing}=nothing,
)
    configure_fast!(oracle.adapter, oracle.model)
    lp_count = 1
    obj, r = solve_oracle!(oracle, h; canonicalize=canonicalize)

    if !supports_two_phase(oracle.adapter)
        configure_default!(oracle.adapter, oracle.model)
        return obj, r, lp_count
    end
    if verify_tol !== nothing && obj >= verify_tol
        configure_default!(oracle.adapter, oracle.model)
        return obj, r, lp_count
    end

    configure_verified!(oracle.adapter, oracle.model)
    obj_verified, r_verified = solve_oracle!(oracle, h; canonicalize=canonicalize)
    lp_count += 1
    configure_default!(oracle.adapter, oracle.model)
    return obj_verified, r_verified, lp_count
end
