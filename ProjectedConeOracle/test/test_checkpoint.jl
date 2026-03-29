using LinearAlgebra
using Test
using ProjectedConeOracle

@testset "Checkpoint" begin
    config = OracleConfig()
    mktempdir() do dir
        file = joinpath(dir, "checkpoint.jls")
        rays = [[1.0, 2.0], [1 / 3, 2 / 3, 1.0]]
        basis = [inv(sqrt(2.0)) 0.0; inv(sqrt(2.0)) 0.0; 0.0 1.0]

        save_checkpoint(file, rays, 7, 11, basis, config)
        rays2, seen, iter, lp_calls, basis2 = load_checkpoint(file, config)

        @test iter == 7
        @test lp_calls == 11
        @test rationalize_ray_coordwise(rays2[1], config) == rationalize_ray_coordwise(rays[1], config)
        @test rationalize_ray_coordwise(rays2[2], config) == rationalize_ray_coordwise(rays[2], config)
        @test seen == Set(hash_key_ray(ray, config) for ray in rays2)
        @test basis2 !== nothing
        @test basis2' * basis2 ≈ Matrix{Float64}(I, size(basis2, 2), size(basis2, 2)) atol=1e-10
    end

    mktempdir() do dir
        bad = joinpath(dir, "corrupt.jls")
        write(bad, "not a checkpoint")
        @test_throws ArgumentError load_checkpoint(bad, config)
    end
end
