using Test
using ProjectedConeOracle

@testset "Facet enumeration" begin
    config = OracleConfig(normaliz_threads=1)

    @testset "orthant_bigint" begin
        rays = BigInt[1 0; 0 1]
        facets = enumerate_facets(rays, config)
        @test size(facets) == (2, 2)
        @test validate_facet(BigInt[1, 0], rays, config)
        @test validate_facet(BigInt[0, 1], rays, config)
        @test !validate_facet(BigInt[-1, 0], rays, config)
    end

    @testset "orthant_float_rationalization" begin
        rays = [0.5 0.0; 0.0 1.5]
        facets = enumerate_facets(rays, config)
        @test size(facets) == (2, 2)
        for i in 1:size(facets, 1)
            @test validate_facet(facets[i, :], rays, config)
        end
    end

    @testset "nonsimplicial_cone" begin
        rays = BigInt[1 0 0; 0 1 0; 0 0 1; 1 1 1]
        facets = enumerate_facets(rays, config)
        @test size(facets) == (3, 3)
        for i in 1:size(facets, 1)
            @test validate_facet(facets[i, :], rays, config)
        end
    end

    @testset "lineality_error" begin
        rays = BigInt[1 0; -1 0]
        @test_throws ErrorException enumerate_facets(rays, config)
    end
end
