using LinearAlgebra
using SparseArrays
using Test
using ProjectedConeOracle

@testset "SeparationOracle" begin
    A2 = sparse(Matrix{Float64}(I, 2, 2))
    c2 = [1.0, 1.0]
    oracle2 = SeparationOracle(A2, [1, 2]; adapter=HiGHSAdapter(), c=c2)

    @testset "orthant2d_min_e1" begin
        obj, y = solve_oracle!(oracle2, [1.0, 0.0]; canonicalize=false)
        @test isapprox(obj, 0.0; atol=1e-8)
        @test length(y) == 2
        @test isapprox(y[1], 0.0; atol=1e-8)
        @test isapprox(y[2], 1.0; atol=1e-8)
    end

    @testset "orthant2d_reuse_model_max_e1" begin
        obj, y = solve_oracle!(oracle2, [-1.0, 0.0]; canonicalize=false)
        @test isapprox(obj, -1.0; atol=1e-8)
        @test isapprox(y[1], 1.0; atol=1e-8)
        @test isapprox(y[2], 0.0; atol=1e-8)
    end

    @testset "projected_index_subset" begin
        A3 = sparse(Matrix{Float64}(I, 3, 3))
        oracle3 = SeparationOracle(A3, [1, 3]; adapter=HiGHSAdapter(), c=[1.0, 1.0])
        obj, y = solve_oracle!(oracle3, [1.0, 0.0]; canonicalize=false)
        @test length(y) == 2
        @test isapprox(obj, 0.0; atol=1e-8)
        @test isapprox(y[1], 0.0; atol=1e-8)
        @test isapprox(y[2], 1.0; atol=1e-8)
    end

    @testset "solve_verified_highs_two_phase" begin
        obj, y, lp_count = solve_verified!(oracle2, [0.0, 1.0]; canonicalize=false, verify_tol=nothing)
        @test isapprox(obj, 0.0; atol=1e-8)
        @test length(y) == 2
        @test lp_count == 2
    end
end
