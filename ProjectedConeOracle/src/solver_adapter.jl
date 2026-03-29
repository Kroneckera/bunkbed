"""
    AbstractSolverAdapter

Abstract type for LP solver backends.  Concrete subtypes: [`HiGHSAdapter`](@ref),
[`GurobiAdapter`](@ref), [`GLPKAdapter`](@ref).
"""
abstract type AbstractSolverAdapter end

"""
    GurobiAdapter(; method=1, crossover=-1, presolve=0, feasibility_tol=1e-9, optimality_tol=1e-9, numeric_focus=0, time_limit=Inf)

Gurobi solver adapter.  Requires `using Gurobi` before use (loaded via package extension).

For this repository's full `enumerate_all_inequalities(3, 4)` pipeline, fresh Round 8 evidence showed:
- `Method=0, Presolve=0` was the fastest correct facet-solve configuration matching the HiGHS ground truth;
- `Method=1, Presolve=0` was the robust default/verified configuration for normalization, seed generation, and verification;
- the mixed public API path is correct when the adapter default stays on `Method=1, Presolve=0` while the fast phase switches to `Method=0, Presolve=0`.

Keyword arguments:
- `method`: `-1`=automatic, `0`=primal simplex, `1`=dual simplex (default), `2`=barrier.
- `crossover`: `-1`=auto (default), `0`=disable, `1`=primal, `2`=dual.
- `presolve`: Gurobi `Presolve` parameter (default `0` for this package's production path).
- `numeric_focus`: 0–3, higher values trade speed for numerical reliability.
"""
Base.@kwdef struct GurobiAdapter <: AbstractSolverAdapter
    method::Int = 1
    crossover::Int = -1
    presolve::Int = 0
    feasibility_tol::Float64 = 1e-9
    optimality_tol::Float64 = 1e-9
    numeric_focus::Int = 0
    time_limit::Float64 = Inf
end

"""
    HiGHSAdapter(; solver="ipm", primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9, time_limit=Inf)

HiGHS solver adapter (always available, default backend).

- `solver`: `"ipm"` for interior-point (default, fast), `"simplex"` for primal/dual simplex.
"""
Base.@kwdef struct HiGHSAdapter <: AbstractSolverAdapter
    solver::String = "ipm"
    primal_feasibility_tolerance::Float64 = 1e-9
    dual_feasibility_tolerance::Float64 = 1e-9
    time_limit::Float64 = Inf
end

"""
    GLPKAdapter(; method=:InteriorPoint, tol_bnd=1e-9, tol_dj=1e-9, time_limit=Inf)

GLPK solver adapter.  Requires `using GLPK` before use (loaded via package extension).

- `method`: `:InteriorPoint` (default) or `:Simplex`.
"""
Base.@kwdef struct GLPKAdapter <: AbstractSolverAdapter
    method::Symbol = :InteriorPoint
    tol_bnd::Float64 = 1e-9
    tol_dj::Float64 = 1e-9
    time_limit::Float64 = Inf
end

# --- Common utilities ---

"""
    supports_two_phase(adapter::AbstractSolverAdapter) -> Bool

Return `true` if `adapter` supports two-phase LP solving (fast barrier + simplex verification).
All built-in adapters return `true`. Override for custom adapters that only support a single
solve method.

See also: [`solve_verified!`](@ref), [`configure_fast!`](@ref), [`configure_verified!`](@ref).
"""
supports_two_phase(::AbstractSolverAdapter) = true

function _set_common_attributes!(model::Model, time_limit::Float64)
    set_silent(model)
    if isfinite(time_limit)
        set_time_limit_sec(model, time_limit)
    end
    return nothing
end

_optimizer(model::Model) = backend(model)

_time_limit(adapter::GurobiAdapter) = adapter.time_limit
_time_limit(adapter::HiGHSAdapter) = adapter.time_limit
_time_limit(adapter::GLPKAdapter) = adapter.time_limit

"""
    create_model(adapter::AbstractSolverAdapter) -> JuMP.Model

Create a new JuMP `Model` backed by `adapter`'s solver, with silent output and the adapter's
default configuration applied.  The model uses `JuMP.direct_model` for zero-overhead access.

See also: [`configure_fast!`](@ref), [`configure_verified!`](@ref).
"""
function create_model(adapter::AbstractSolverAdapter)::Model
    model = direct_model(optimizer_factory(adapter)())
    _set_common_attributes!(model, _time_limit(adapter))
    configure_default!(adapter, model)
    return model
end

# --- HiGHS adapter (always available) ---

"""
    optimizer_factory(adapter::AbstractSolverAdapter) -> Function

Return a zero-argument callable that creates the underlying MathOptInterface optimizer for
`adapter`.  Override this method when implementing a custom solver adapter.
"""
optimizer_factory(::HiGHSAdapter) = () -> HiGHS.Optimizer()

function _apply_highs!(adapter::HiGHSAdapter, model::Model, solver::String)
    set_optimizer_attribute(model, "solver", solver)
    set_optimizer_attribute(model, "primal_feasibility_tolerance", adapter.primal_feasibility_tolerance)
    set_optimizer_attribute(model, "dual_feasibility_tolerance", adapter.dual_feasibility_tolerance)
    return nothing
end

"""
    configure_fast!(adapter::AbstractSolverAdapter, model::Model) -> Nothing

Configure `model` for fast (barrier/interior-point) solving. Used as the first phase of
two-phase verification in [`solve_verified!`](@ref).

See also: [`configure_verified!`](@ref), [`configure_recovery!`](@ref).
"""
function configure_fast!(adapter::HiGHSAdapter, model::Model)::Nothing
    _apply_highs!(adapter, model, "ipm")
    return nothing
end

"""
    configure_verified!(adapter::AbstractSolverAdapter, model::Model) -> Nothing

Configure `model` for verified (simplex) solving. Used as the second phase of two-phase
verification in [`solve_verified!`](@ref) to obtain a basic feasible solution.

See also: [`configure_fast!`](@ref), [`configure_recovery!`](@ref).
"""
function configure_verified!(adapter::HiGHSAdapter, model::Model)::Nothing
    _apply_highs!(adapter, model, "simplex")
    return nothing
end

"""
    configure_recovery!(adapter::AbstractSolverAdapter, model::Model) -> Nothing

Configure `model` for recovery solving after a `NUMERICAL_ERROR` status. Typically uses
simplex with maximum numeric focus to attempt a clean solve.

See also: [`configure_fast!`](@ref), [`configure_verified!`](@ref).
"""
function configure_recovery!(adapter::HiGHSAdapter, model::Model)::Nothing
    _apply_highs!(adapter, model, "simplex")
    return nothing
end

"""
    configure_default!(adapter::AbstractSolverAdapter, model::Model) -> Nothing

Restore `model` to `adapter`'s default solver configuration. Called after two-phase
verification or recovery to reset the model state.
"""
function configure_default!(adapter::HiGHSAdapter, model::Model)::Nothing
    _apply_highs!(adapter, model, adapter.solver)
    return nothing
end

# --- Default solver selection ---

"""
    default_solver_adapter() -> AbstractSolverAdapter

Return the default LP solver adapter, currently `HiGHSAdapter()`.
"""
function default_solver_adapter()::AbstractSolverAdapter
    return HiGHSAdapter()
end
