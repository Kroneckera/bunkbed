using PercolationOracle
using Test

include(joinpath(@__DIR__, "test_partitions.jl"))
include(joinpath(@__DIR__, "test_packed_state.jl"))
include(joinpath(@__DIR__, "test_colorings.jl"))
include(joinpath(@__DIR__, "test_transitions.jl"))
include(joinpath(@__DIR__, "test_oracle.jl"))
include(joinpath(@__DIR__, "test_regression.jl"))
include(joinpath(@__DIR__, "test_threaded.jl"))
include(joinpath(@__DIR__, "test_lp_pipeline.jl"))
include(joinpath(@__DIR__, "test_inequality_enumeration.jl"))
