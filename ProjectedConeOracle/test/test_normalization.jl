using LinearAlgebra
using SparseArrays
using Test
using ProjectedConeOracle

@testset "Dual normalization" begin
    config = OracleConfig()
    adapter = HiGHSAdapter()

    @testset "orthant_2d" begin
        A = sparse(Matrix{Float64}(I, 2, 2))
        c, seeds = find_normalization_vector(A, [1, 2], config, adapter)
        @test isapprox(c, [1.0, 1.0]; atol=1e-8)
        @test length(seeds) == 2
        seed_keys = Set(hash_key_ray(seed, config) for seed in seeds)
        @test hash_key_ray([1.0, 0.0], config) in seed_keys
        @test hash_key_ray([0.0, 1.0], config) in seed_keys
    end

    @testset "projected_subset" begin
        A = sparse(Matrix{Float64}(I, 3, 3))
        c, seeds = find_normalization_vector(A, [1, 3], config, adapter)
        @test isapprox(c, [1.0, 1.0]; atol=1e-8)
        @test length(seeds) == 2
        @test all(length(seed) == 2 for seed in seeds)
    end

    @testset "compatibility_wrapper" begin
        A = sparse(Matrix{Float64}(I, 2, 2))
        c, seeds = find_dual_normalization_vector(A, [1, 2]; adapter=adapter, config=config)
        @test isapprox(c, [1.0, 1.0]; atol=1e-8)
        @test !isempty(seeds)
    end

    @testset "degenerate_zero_projection" begin
        A = spzeros(Float64, 1, 2)
        @test_throws ErrorException find_normalization_vector(A, [1], config, adapter)
    end
end
