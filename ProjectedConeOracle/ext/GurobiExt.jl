module GurobiExt

using ProjectedConeOracle
import ProjectedConeOracle: optimizer_factory, configure_fast!, configure_verified!,
    configure_recovery!, configure_default!, _optimizer
using JuMP
import Gurobi

function ProjectedConeOracle.optimizer_factory(::GurobiAdapter)
    return () -> Gurobi.Optimizer()
end

function _apply_gurobi!(adapter::GurobiAdapter, model::Model; method::Int, crossover::Int, presolve::Int, numeric_focus::Int)
    set_optimizer_attribute(model, "Method", method)
    set_optimizer_attribute(model, "Crossover", crossover)
    set_optimizer_attribute(model, "Presolve", presolve)
    set_optimizer_attribute(model, "FeasibilityTol", adapter.feasibility_tol)
    set_optimizer_attribute(model, "OptimalityTol", adapter.optimality_tol)
    set_optimizer_attribute(model, "NumericFocus", numeric_focus)
    return nothing
end

function ProjectedConeOracle.configure_fast!(adapter::GurobiAdapter, model::Model)::Nothing
    # Round 8 full-pipeline matrix: Method=0, Presolve=0 was the fastest uniform row
    # matching the HiGHS 17-ray / 19-facet ground truth.
    _apply_gurobi!(adapter, model; method=0, crossover=-1, presolve=0, numeric_focus=adapter.numeric_focus)
    return nothing
end

function ProjectedConeOracle.configure_verified!(adapter::GurobiAdapter, model::Model)::Nothing
    # Verified phase stays on presolve-off dual simplex: it matched the HiGHS ground
    # truth in the Round 8 matrix and remains the most conservative production check.
    _apply_gurobi!(adapter, model; method=1, crossover=-1, presolve=0, numeric_focus=adapter.numeric_focus)
    return nothing
end

function ProjectedConeOracle.configure_recovery!(adapter::GurobiAdapter, model::Model)::Nothing
    _apply_gurobi!(adapter, model; method=1, crossover=-1, presolve=0, numeric_focus=3)
    return nothing
end

function ProjectedConeOracle.configure_default!(adapter::GurobiAdapter, model::Model)::Nothing
    _apply_gurobi!(adapter, model; method=adapter.method, crossover=adapter.crossover, presolve=adapter.presolve, numeric_focus=adapter.numeric_focus)
    return nothing
end

end # module GurobiExt
