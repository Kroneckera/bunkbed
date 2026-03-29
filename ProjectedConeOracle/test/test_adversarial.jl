using LinearAlgebra
using SparseArrays
using Test
using ProjectedConeOracle

function _direction_error(z::AbstractVector{<:Integer}, ray::AbstractVector{<:Real})
    zf = Float64.(collect(z))
    return norm(zf ./ norm(zf) - ray ./ norm(ray))
end

function _captured_error(f::Function)
    try
        f()
        return nothing
    catch err
        return sprint(showerror, err)
    end
end

@testset "Adversarial edge cases" begin

    @testset "T1: Overflower – near-irrational denominator stress" begin
        ray = [1.0, inv(pi), 1 / (10^9 + 7)]

        small_config = OracleConfig(max_den=10^6)
        large_config = OracleConfig(max_den=10^12)

        z_small = rationalize_ray_coordwise(ray, small_config)
        z_large = rationalize_ray_coordwise(ray, large_config)

        @test z_small[1] > 0
        @test z_small[2] > 0
        @test z_small[3] == 0

        @test z_large[1] > 0
        @test z_large[2] > 0
        @test z_large[3] > 0

        err_small = _direction_error(z_small, ray)
        err_large = _direction_error(z_large, ray)
        @test err_large < err_small
    end

    @testset "T1: Overflower – BigInt matrix path avoids Int64 overflow" begin
        # Pairwise coprime denominators near 10^6 force the common denominator
        # well beyond Int64 when the exact direction is lifted to primitive integers.
        overflow_ray = [1.0, 1 / 999_983, 1 / 999_979, 1 / 999_971, 1 / 999_961]
        overflow_config = OracleConfig(max_den=10^6)

        M = ProjectedConeOracle._rationalized_ray_matrix([overflow_ray], overflow_config)
        @test eltype(M) == BigInt
        @test size(M) == (1, 5)
        @test maximum(abs.(M)) > BigInt(typemax(Int64))
        @test ProjectedConeOracle._rationalize_ray_i64(overflow_ray, overflow_config) === nothing

        err_matrix = _direction_error(vec(M[1, :]), overflow_ray)
        @test err_matrix < 1e-6
    end

    @testset "T2: Flat Cone – maybe_expand! tolerance sensitivity" begin
        # A cone in R³ that is nearly flat in the z-direction.
        # Two clear dimensions (x,y) and a z-thickness of ~1e-7.
        # The key question: does maybe_expand! detect the thin z-dimension?

        # Start with xy-plane basis (2D subspace of R³)
        sb_loose = SubspaceBasis(hcat([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
        sb_tight = SubspaceBasis(hcat([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))

        # A ray from the flat cone: large x,y components, tiny z = 1e-7
        # Perpendicular component to the xy-basis has norm 1e-7
        flat_ray = [0.6, 0.8, 1e-7]

        # Loose config: base_eps=1e-7 → quantization_eps = 1e-6 > 1e-7
        # maybe_expand! treats the z-component as noise → dimension lost
        loose_config = OracleConfig(base_eps=1e-7)
        expanded_loose = maybe_expand!(sb_loose, flat_ray, loose_config)
        @test !expanded_loose
        @test size(sb_loose.B, 2) == 2

        # Tight config: base_eps=1e-10 → quantization_eps = 1e-9 < 1e-7
        # maybe_expand! detects the z-component → full 3D
        tight_config = OracleConfig(base_eps=1e-10)
        expanded_tight = maybe_expand!(sb_tight, flat_ray, tight_config)
        @test expanded_tight
        @test size(sb_tight.B, 2) == 3
    end

    @testset "T2: Flat Cone – SeparationOracle integration" begin
        # Cone: x1 >= 0, x2 >= 0, x3 >= 0, x1 + x2 - 1e7*x3 >= 0
        # Forces x3 <= (x1+x2)/1e7, making the cone flat in z.
        A = sparse([
            1.0  0.0  0.0;
            0.0  1.0  0.0;
            0.0  0.0  1.0;
            1.0  1.0 -1e7;
        ])
        J = [1, 2, 3]
        c = [1.0, 1.0, 1.0]

        oracle = SeparationOracle(A, J; adapter=HiGHSAdapter(), c=c)

        # Minimize -y3 (maximize y3) to find the ray with largest z-component
        obj, y = solve_oracle!(oracle, [0.0, 0.0, -1.0]; canonicalize=false)

        # LP optimum: y3 = 1/(1e7+1) ≈ 1e-7, y1+y2 ≈ 1
        @test y[3] > 0
        @test y[3] < 1e-5

        # Feed this oracle-produced ray through maybe_expand!
        sb_loose = SubspaceBasis(hcat([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
        sb_tight = SubspaceBasis(hcat([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))

        loose_config = OracleConfig(base_eps=1e-7)   # quantization_eps = 1e-6
        tight_config = OracleConfig(base_eps=1e-10)   # quantization_eps = 1e-9

        # After canonical_ray, the max component is ~0.5-1.0, so z-component
        # in canonical form is ~2e-7. Perpendicular norm to xy-plane is ~2e-7.
        y_canon = canonical_ray(y)

        expanded_loose = maybe_expand!(sb_loose, y_canon, loose_config)
        expanded_tight = maybe_expand!(sb_tight, y_canon, tight_config)

        # Loose tolerance misses the thin dimension
        @test !expanded_loose
        @test size(sb_loose.B, 2) == 2

        # Tight tolerance catches it
        @test expanded_tight
        @test size(sb_tight.B, 2) == 3
    end

    @testset "T3: Perturbed Ray – rationalization with small max_den" begin
        # A ray requiring denominator 10^9 to represent exactly.
        # canonical_ray([1.0, 1e-9]) = [1.0, 1e-9] (max-abs normalized, both > canonical_tol)
        ray = [1.0, 1e-9]

        # Small max_den (10^7): denominator 10^9 exceeds limit.
        # Tolerance doubling eventually rounds 1e-9 to 0.
        small_config = OracleConfig(max_den=10_000_000)
        z_small = rationalize_ray_coordwise(ray, small_config)

        @test length(z_small) == 2
        @test !all(iszero, z_small)       # At least first coordinate nonzero
        @test z_small[1] != BigInt(0)      # x-direction preserved
        @test z_small[2] == BigInt(0)      # y-direction lost (rounded to 0)
    end

    @testset "T3: Perturbed Ray – rationalization with large max_den" begin
        ray = [1.0, 1e-9]

        # Large max_den (10^10): can represent 1/10^9 exactly.
        large_config = OracleConfig(max_den=10_000_000_000)
        z_large = rationalize_ray_coordwise(ray, large_config)

        @test length(z_large) == 2
        @test z_large[1] > BigInt(0)       # x-direction preserved
        @test z_large[2] > BigInt(0)       # y-direction preserved (not rounded away)

        # The ratio z[1]/z[2] should be ≈ 10^9
        ratio = Float64(z_large[1]) / Float64(z_large[2])
        @test isapprox(ratio, 1e9; rtol=0.15)
    end

    @testset "T3: Perturbed Ray – accuracy comparison" begin
        ray = [1.0, 1e-9]

        small_config = OracleConfig(max_den=10_000_000)
        large_config = OracleConfig(max_den=10_000_000_000)

        z_small = rationalize_ray_coordwise(ray, small_config)
        z_large = rationalize_ray_coordwise(ray, large_config)

        # True direction (normalized)
        true_dir = ray ./ norm(ray)

        # Large-denom result: both components nonzero, close to true direction
        dir_large = Float64.(z_large) ./ norm(Float64.(z_large))
        err_large = norm(dir_large - true_dir)
        @test err_large < 1e-6

        # Small-denom result: second component is 0, so direction is [1,0]
        dir_small = Float64.(z_small) ./ norm(Float64.(z_small))
        err_small = norm(dir_small - true_dir)

        # Large denominator is strictly more accurate
        @test err_large < err_small
    end

    @testset "T4: Silent Lineality – compute_cone reports lineality" begin
        config = OracleConfig(normaliz_threads=1)
        G = BigInt[1 0 0; -1 0 0; 0 1 0; 0 0 1]

        result = compute_cone(NormalizCLIBackend(), G, config)

        @test !is_pointed(result)
        @test size(result.lineality_space) == (1, 3)
        @test abs(result.lineality_space[1, 1]) == 1
        @test result.lineality_space[1, 2] == 0
        @test result.lineality_space[1, 3] == 0
    end

    @testset "T4: Silent Lineality – projected_cone_oracle rejects non-pointed projections" begin
        A = sparse([1.0 0.0 0.0])
        configs = (
            OracleConfig(normaliz_threads=1, shuffle_facets=false, base_eps=1e-7),
            OracleConfig(normaliz_threads=1, shuffle_facets=false, base_eps=1e-10),
        )

        for cfg in configs
            err = _captured_error() do
                projected_cone_oracle(A, [1, 2], cfg, HiGHSAdapter())
            end
            @test err !== nothing
            @test occursin("lineality", lowercase(err))
            @test occursin("dual_infeasible", lowercase(err)) ||
                  occursin("non-pointed", lowercase(err))
        end
    end

end
