using Test
using ProjectedConeOracle

@testset "ProjectedConeOracle" begin
    include("test_canonicalization.jl")
    include("test_rationalization.jl")
    include("test_subspace.jl")
    include("test_checkpoint.jl")
    include("test_solver_adapter.jl")
    include("separation_oracle.jl")
    include("test_normaliz_cli.jl")
    include("test_facet_enumeration.jl")
    include("test_normalization.jl")
    include("test_adversarial.jl")
    include("test_projection_algorithm.jl")
end
