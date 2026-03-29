using Test
using ProjectedConeOracle

function _rowset(M)
    return Set(Tuple(M[i, :]) for i in 1:size(M, 1))
end

function _write_fixture(path::AbstractString, body::AbstractString)
    write(path, replace(body, "\n    " => "\n"))
    return path
end

@testset "Normaliz CLI backend" begin
    config = OracleConfig(normaliz_threads=1)
    backend = NormalizCLIBackend()

    @testset "parser_prefers_modern_cst_and_msp" begin
        mktempdir() do dir
            base = joinpath(dir, "cone")
            _write_fixture(
                base * ".cst",
                """
                3
                3
                0 0 1
                0 1 0
                1 0 0
                inequalities
                0
                3
                equations
                0
                4
                congruences
                1
                3
                1 1 1
                grading
                """,
            )
            _write_fixture(
                base * ".esp",
                """
                1
                3
                9 9 9
                inequalities
                """,
            )
            _write_fixture(
                base * ".msp",
                """
                1
                3
                1 0 0
                """,
            )
            _write_fixture(
                base * ".out",
                """
                1 basis elements of maximal subspace:
                 0 1 0

                1 support hyperplanes:
                 8 8 8
                """,
            )

            support = ProjectedConeOracle._parse_support_hyperplanes(base, 3)
            lineality = ProjectedConeOracle._parse_lineality_space(base, 3)

            @test _rowset(support) == Set([(BigInt(0), BigInt(0), BigInt(1)),
                                           (BigInt(0), BigInt(1), BigInt(0)),
                                           (BigInt(1), BigInt(0), BigInt(0))])
            @test _rowset(lineality) == Set([(BigInt(1), BigInt(0), BigInt(0))])
        end
    end

    @testset "parser_legacy_sup_and_esp_fallbacks" begin
        mktempdir() do dir
            sup_base = joinpath(dir, "supcone")
            _write_fixture(
                sup_base * ".sup",
                """
                2
                3
                1 0 0
                0 1 0
                """,
            )
            support_sup = ProjectedConeOracle._parse_support_hyperplanes(sup_base, 3)
            @test _rowset(support_sup) == Set([(BigInt(1), BigInt(0), BigInt(0)),
                                               (BigInt(0), BigInt(1), BigInt(0))])

            esp_base = joinpath(dir, "espcone")
            _write_fixture(
                esp_base * ".esp",
                """
                2
                3
                0 0 1
                0 1 0
                inequalities
                """,
            )
            support_esp = ProjectedConeOracle._parse_support_hyperplanes(esp_base, 3)
            @test _rowset(support_esp) == Set([(BigInt(0), BigInt(0), BigInt(1)),
                                               (BigInt(0), BigInt(1), BigInt(0))])
        end
    end

    @testset "parser_out_fallbacks" begin
        mktempdir() do dir
            base = joinpath(dir, "legacy")
            _write_fixture(
                base * ".out",
                """
                2 extreme rays:
                 1 0 0
                 0 1 0

                1 basis elements of maximal subspace:
                 0 0 1

                2 support hyperplanes:
                 1 0 0
                 0 1 0
                """,
            )

            support = ProjectedConeOracle._parse_support_hyperplanes(base, 3)
            rays = ProjectedConeOracle._parse_extreme_rays(base, 3)
            lineality = ProjectedConeOracle._parse_lineality_space(base, 3)

            @test _rowset(support) == Set([(BigInt(1), BigInt(0), BigInt(0)),
                                           (BigInt(0), BigInt(1), BigInt(0))])
            @test _rowset(rays) == Set([(BigInt(1), BigInt(0), BigInt(0)),
                                        (BigInt(0), BigInt(1), BigInt(0))])
            @test _rowset(lineality) == Set([(BigInt(0), BigInt(0), BigInt(1))])
        end
    end

    @testset "orthant_2d" begin
        G = BigInt[1 0; 0 1]
        result = compute_cone(backend, G, config)
        @test is_pointed(result)
        @test eltype(result.support_hyperplanes) == BigInt
        @test eltype(result.extreme_rays) == BigInt
        @test _rowset(result.support_hyperplanes) == Set([(BigInt(1), BigInt(0)), (BigInt(0), BigInt(1))])
        @test _rowset(result.extreme_rays) == Set([(BigInt(1), BigInt(0)), (BigInt(0), BigInt(1))])
        @test size(result.lineality_space) == (0, 2)
    end

    @testset "orthant_3d" begin
        G = BigInt[1 0 0; 0 1 0; 0 0 1]
        result = compute_cone(backend, G, config)
        @test is_pointed(result)
        @test size(result.support_hyperplanes) == (3, 3)
        @test size(result.extreme_rays) == (3, 3)
    end

    @testset "single_ray" begin
        G = BigInt[1 2 3]
        result = compute_cone(backend, G, config)
        @test is_pointed(result)
        @test size(result.extreme_rays) == (1, 3)
        @test Tuple(result.extreme_rays[1, :]) == (BigInt(1), BigInt(2), BigInt(3))
    end

    @testset "lineality_detection" begin
        G = BigInt[1 0 0; -1 0 0; 0 1 0]
        result = compute_cone(backend, G, config)
        @test !is_pointed(result)
        @test size(result.lineality_space) == (1, 3)
        @test abs(result.lineality_space[1, 1]) == 1
        @test result.lineality_space[1, 2] == 0
        @test result.lineality_space[1, 3] == 0
    end

    @testset "empty_generators" begin
        G = Matrix{BigInt}(undef, 0, 3)
        result = compute_cone(backend, G, config)
        @test is_pointed(result)
        @test size(result.support_hyperplanes) == (0, 3)
        @test size(result.extreme_rays) == (0, 3)
        @test size(result.lineality_space) == (0, 3)
    end
end
