using LinearAlgebra
using SparseArrays
using Test
using ProjectedConeOracle

_rayset(vs, config=OracleConfig()) = Set(hash_key_ray(v, config) for v in vs)
_facetset(vs, config=OracleConfig()) = Set(hash_key_normal(v, config) for v in vs)

@testset "Projection algorithm" begin
    config = OracleConfig(normaliz_threads=1, shuffle_facets=false)
    adapter = HiGHSAdapter()

    @testset "orthant_2d" begin
        A = sparse(Matrix{Float64}(I, 2, 2))
        result = projected_cone_oracle(A, [1, 2], config, adapter)

        @test result.converged
        @test result.iterations >= 1
        @test result.lp_calls > 0
        @test _rayset(result.rays, config) == Set([
            hash_key_ray([1.0, 0.0], config),
            hash_key_ray([0.0, 1.0], config),
        ])
        @test _facetset(result.facets, config) == Set([
            hash_key_normal([1.0, 0.0], config),
            hash_key_normal([0.0, 1.0], config),
        ])
    end

    @testset "orthant_3d_projected_to_2d_wrapper" begin
        A = Matrix{Float64}(I, 3, 3)
        result = projected_cone(A, [1, 3]; config=config, adapter=adapter)

        @test result.converged
        @test _rayset(result.rays, config) == Set([
            hash_key_ray([1.0, 0.0], config),
            hash_key_ray([0.0, 1.0], config),
        ])
        @test _facetset(result.facets, config) == Set([
            hash_key_normal([1.0, 0.0], config),
            hash_key_normal([0.0, 1.0], config),
        ])
    end

    @testset "baseline_diagonal_ray_example" begin
        A = sparse([
            1.0 -1.0
           -1.0  1.0
            1.0  1.0
        ])
        result = projected_cone_oracle(A, [1, 2], config, adapter)

        @test result.converged
        @test result.iterations >= 1
        @test length(result.rays) == 1
        @test _rayset(result.rays, config) == Set([hash_key_ray([1.0, 1.0], config)])
        @test length(result.facets) == 1
        @test _facetset(result.facets, config) == Set([hash_key_normal([1.0, 1.0], config)])
    end

    @testset "oracle facet normals are scale invariant" begin
        sb = ProjectedConeOracle.SubspaceBasis(Matrix{Float64}(I, 2, 2))
        h = ProjectedConeOracle._oracle_normal(sb, [2.0, -4.0], config)
        h_scaled = ProjectedConeOracle._oracle_normal(sb, 1.0e40 .* [2.0, -4.0], config)

        @test h == [0.5, -1.0]
        @test h_scaled == h
        @test maximum(abs, h_scaled) == 1.0
    end

    @testset "checkpoint_resume" begin
        mktempdir() do dir
            checkpoint = joinpath(dir, "orthant_checkpoint.jls")
            cfg_ckpt = OracleConfig(normaliz_threads=1, shuffle_facets=false, checkpoint_file=checkpoint)
            A = sparse(Matrix{Float64}(I, 2, 2))

            first_run = projected_cone_oracle(A, [1, 2], cfg_ckpt, adapter)
            second_run = projected_cone_oracle(A, [1, 2], cfg_ckpt, adapter)

            @test first_run.converged
            @test second_run.converged
            @test _rayset(first_run.rays, cfg_ckpt) == _rayset(second_run.rays, cfg_ckpt)
            @test _facetset(first_run.facets, cfg_ckpt) == _facetset(second_run.facets, cfg_ckpt)
        end
    end
end
