module GLPKExt

using ProjectedConeOracle
import ProjectedConeOracle: optimizer_factory, configure_fast!, configure_verified!,
    configure_recovery!, configure_default!, _optimizer
using JuMP
import GLPK

_glpk_method(method::Symbol) = method == :InteriorPoint ? GLPK.INTERIOR : GLPK.SIMPLEX

function ProjectedConeOracle.optimizer_factory(adapter::GLPKAdapter)
    return () -> GLPK.Optimizer(method=_glpk_method(adapter.method))
end

function _apply_glpk!(adapter::GLPKAdapter, model::Model, method::Symbol)
    _optimizer(model).method = _glpk_method(method)
    set_optimizer_attribute(model, "tol_bnd", adapter.tol_bnd)
    set_optimizer_attribute(model, "tol_dj", adapter.tol_dj)
    return nothing
end

function ProjectedConeOracle.configure_fast!(adapter::GLPKAdapter, model::Model)::Nothing
    _apply_glpk!(adapter, model, :InteriorPoint)
    return nothing
end

function ProjectedConeOracle.configure_verified!(adapter::GLPKAdapter, model::Model)::Nothing
    _apply_glpk!(adapter, model, :Simplex)
    return nothing
end

function ProjectedConeOracle.configure_recovery!(adapter::GLPKAdapter, model::Model)::Nothing
    _apply_glpk!(adapter, model, :Simplex)
    return nothing
end

function ProjectedConeOracle.configure_default!(adapter::GLPKAdapter, model::Model)::Nothing
    _apply_glpk!(adapter, model, adapter.method)
    return nothing
end

end # module GLPKExt
