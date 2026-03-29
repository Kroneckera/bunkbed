"""
    gcd_vec(z::AbstractVector{<:Integer}) -> Integer

Compute the GCD of all elements in `z`. Returns `0` for an all-zero vector.

See also: [`rationalize_ray_coordwise`](@ref).
"""
function gcd_vec(z::AbstractVector{<:Integer})
    g = zero(eltype(z))
    for zi in z
        g = g == 0 ? abs(zi) : gcd(g, abs(zi))
    end
    return g
end

"""
    rationalize_ray_coordwise(r, config=OracleConfig()) -> Vector{BigInt}

Convert a floating-point ray `r` to a primitive integer ray via coordinate-wise rational
approximation. The ray is first canonicalized, then each coordinate is rationalized with
tolerance starting at `config.rat_tol` (doubling up to `config.rat_tol_ceiling` if the
denominator exceeds `config.max_den`). The result is scaled by the LCM of denominators and
divided by the GCD to produce a primitive integer vector.

Errors if the input ray is numerically zero after canonicalization.

See also: [`gcd_vec`](@ref), [`canonical_ray`](@ref).
"""
function rationalize_ray_coordwise(r::AbstractVector{<:Real}, config::OracleConfig=OracleConfig())
    rh = canonical_ray(r, config)
    all(iszero, rh) && error("rationalize_ray_coordwise: input ray is (numerically) zero")

    qs = Vector{Rational{BigInt}}(undef, length(rh))
    dens = Vector{BigInt}(undef, length(rh))
    max_den = BigInt(config.max_den)

    for i in eachindex(rh)
        tol2 = config.rat_tol
        qi = rationalize(BigInt, rh[i]; tol=tol2)
        while denominator(qi) > max_den && tol2 < config.rat_tol_ceiling
            tol2 = min(tol2 * 2, config.rat_tol_ceiling)
            qi = rationalize(BigInt, rh[i]; tol=tol2)
        end
        qs[i] = qi
        dens[i] = denominator(qi)
    end

    Q = dens[1]
    for i in 2:length(dens)
        Q = lcm(Q, dens[i])
    end

    z = Vector{BigInt}(undef, length(rh))
    for i in eachindex(rh)
        z[i] = Q * numerator(qs[i]) ÷ denominator(qs[i])
    end

    g = gcd_vec(z)
    if g > 1
        z .÷= g
    end
    return z
end

function _rationalize_ray_i64(r::AbstractVector{<:Real}, config::OracleConfig=OracleConfig())
    z = rationalize_ray_coordwise(r, config)
    out = Vector{Int64}(undef, length(z))
    for i in eachindex(z)
        zi = z[i]
        if zi > typemax(Int64) || zi < typemin(Int64)
            return nothing
        end
        out[i] = Int64(zi)
    end
    return out
end
