using LinearAlgebra
using Test
using ProjectedConeOracle

@testset "Subspace" begin
    config = OracleConfig()

    sb_full = SubspaceBasis(Matrix{Float64}(I, 3, 3))
    y = [0.2, -0.4, 1.0]
    @test lift(sb_full, project(sb_full, y)) ≈ y

    B = [inv(sqrt(2.0)) 0.0; inv(sqrt(2.0)) 0.0; 0.0 1.0]
    sb = SubspaceBasis(B)
    y2 = [2.0, 0.0, 3.0]
    @test lift(sb, project(sb, y2)) ≈ [1.0, 1.0, 3.0]

    sb_expand = SubspaceBasis(reshape([1.0, 0.0], 2, 1))
    @test maybe_expand!(sb_expand, [0.0, 1.0], config)
    @test size(sb_expand.B, 2) == 2

    sb_in_span = SubspaceBasis(Matrix{Float64}(I, 2, 2))
    @test !maybe_expand!(sb_in_span, [0.5, 0.5], config)

    z = [2.0, 3.0]
    c = [1.0, 1.0]
    w = cy_normalize(z, c, config)
    @test isapprox(dot(c, w), 1.0; atol=1e-12)

    tight = OracleConfig(cy_normalize_tol=1e-8)
    @test_throws ErrorException cy_normalize([1.0, 0.0], [1e-9, 0.0], tight)
end
