module ProjectedConeOracle

using LinearAlgebra
using Serialization
using SparseArrays
using Random
using JuMP
import MathOptInterface as MOI
using HiGHS

include("types.jl")
include("config.jl")
include("solver_adapter.jl")
include("canonicalization.jl")
include("subspace.jl")
include("rationalization.jl")
include("checkpoint.jl")
include("normaliz_cli.jl")
include("facet_enumeration.jl")
include("SeparationOracle.jl")
include("dual_normalization.jl")
include("ProjectionAlgorithm.jl")

export OracleConfig
export ProjectedConeResult
export AbstractSolverAdapter, GurobiAdapter, HiGHSAdapter, GLPKAdapter
export create_model, configure_fast!, configure_verified!, configure_recovery!, configure_default!
export optimizer_factory, supports_two_phase, default_solver_adapter
export SubspaceBasis, project, lift, maybe_expand!, cy_normalize
export canonical_ray, canonical_normal, hash_key_ray, hash_key_normal
export rationalize_ray_coordwise, gcd_vec
export save_checkpoint, load_checkpoint
export AbstractNormalizBackend, NormalizCLIBackend, NormalizResult
export compute_cone, is_pointed, enumerate_facets, validate_facet
export SeparationOracle
export set_solver_method!, solve_oracle!, solve_verified!
export find_normalization_vector, find_dual_normalization_vector
export projected_cone_oracle, projected_cone
export normaliz_backend_status

end
