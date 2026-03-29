using LinearAlgebra: dot
using Test
using ProjectedConeOracle

@testset "Rationalization" begin
    config = OracleConfig()
    @test rationalize_ray_coordwise([0.5, 1.0], config) == BigInt[1, 2]
    @test rationalize_ray_coordwise([1 / 3, 2 / 3, 1.0], config) == BigInt[1, 2, 3]
    @test gcd_vec(BigInt[6, 9, 15]) == BigInt(3)
    @test gcd_vec(BigInt[0, 0, 0]) == BigInt(0)

    r = [0.2, 0.4, 1.0]
    z = rationalize_ray_coordwise(r, config)
    @test dot(Float64.(z), r) > 0

    config_overflow = OracleConfig(rat_tol=1e-30, max_den=typemax(Int))
    overflow_ray = [1.0, 0.09913970137863681, 0.7019797138879542]
    @test ProjectedConeOracle._rationalize_ray_i64(overflow_ray, config_overflow) === nothing
end
