using Test
using ProjectedConeOracle

@testset "Canonicalization" begin
    @testset "OracleConfig validation" begin
        cfg = OracleConfig(base_eps=1e-11)
        @test isapprox(cfg.canonical_tol, 1e-14; atol=1e-18, rtol=0.0)
        @test isapprox(cfg.quantization_eps, 1e-10; atol=1e-18, rtol=0.0)
        @test cfg.seed_tol == cfg.violation_tol
        cfg_alias = OracleConfig(max_iterations=7)
        @test cfg_alias.max_rounds == 7
        @test cfg_alias.max_iterations == 7
        @test_throws ArgumentError OracleConfig(seed_tol=1e-7, violation_tol=1e-6)
        @test_throws ArgumentError OracleConfig(max_den=0)
        @test_throws ArgumentError OracleConfig(max_rounds=3, max_iterations=4)
    end

    config = OracleConfig()
    @test canonical_ray([2.0, 4.0, 6.0], config) ≈ [1 / 3, 2 / 3, 1.0]
    @test canonical_ray([0.0, 0.0, 0.0], config) == zeros(3)
    @test canonical_normal([2.0, 4.0, 6.0], config) ≈ [1 / 3, 2 / 3, 1.0]

    r = [0.25, 0.5]
    @test hash_key_ray(r, config) == hash_key_ray(2 .* r, config)
    @test hash_key_ray(r, config) != hash_key_ray(-1 .* r, config)

    c = canonical_ray([3.0, 6.0, 9.0], config)
    @test canonical_ray(c, config) == c

    cfg_fine = OracleConfig(quantization_eps=1e-10)
    cfg_coarse = OracleConfig(quantization_eps=1e-6)
    plus = [1.0, 1.0 + 1e-9]
    minus = [1.0, 1.0 - 1e-9]
    @test hash_key_ray(plus, cfg_fine) != hash_key_ray(minus, cfg_fine)
    @test hash_key_ray(plus, cfg_coarse) == hash_key_ray(minus, cfg_coarse)
end
