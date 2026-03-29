"""
    SubspaceBasis

Orthonormal basis for a working subspace of the projected cone's ambient space.
Manages dimension expansion as new rays are discovered outside the current span.

Construct with `SubspaceBasis(B::Matrix)` from an explicit basis matrix, or
`SubspaceBasis(d::Integer)` for the full `d`-dimensional identity basis.

See also: [`project`](@ref), [`lift`](@ref), [`maybe_expand!`](@ref).
"""
mutable struct SubspaceBasis
    B::Matrix{Float64}
end

SubspaceBasis(B::AbstractMatrix{<:Real}) = SubspaceBasis(Matrix{Float64}(B))
SubspaceBasis(d::Integer) = SubspaceBasis(Matrix{Float64}(I, d, d))

"""
    project(sb::SubspaceBasis, y::AbstractVector{<:Real}) -> Vector{Float64}

Project ambient-space vector `y` into the reduced coordinates: `z = B' y`.

See also: [`lift`](@ref), [`SubspaceBasis`](@ref).
"""
project(sb::SubspaceBasis, y::AbstractVector{<:Real}) = sb.B' * Vector{Float64}(y)

"""
    lift(sb::SubspaceBasis, z::AbstractVector{<:Real}) -> Vector{Float64}

Lift reduced-coordinate vector `z` back to the ambient space: `y = B z`.

See also: [`project`](@ref), [`SubspaceBasis`](@ref).
"""
lift(sb::SubspaceBasis, z::AbstractVector{<:Real}) = sb.B * Vector{Float64}(z)

is_full_rank(sb::SubspaceBasis) = size(sb.B, 1) == size(sb.B, 2)
reduced_dim(sb::SubspaceBasis) = size(sb.B, 2)
ambient_dim(sb::SubspaceBasis) = size(sb.B, 1)

function reorthogonalize_basis(B::AbstractMatrix{<:Real})
    BF = Matrix{Float64}(B)
    size(BF, 2) == 0 && return zeros(Float64, size(BF, 1), 0)
    F = qr(BF)
    return Matrix(F.Q)[:, 1:size(BF, 2)]
end

"""
    maybe_expand!(sb::SubspaceBasis, y, config=OracleConfig(); tol=nothing) -> Bool

If `y` has a component perpendicular to the current basis larger than `tol` (default:
`config.quantization_eps`), extend the basis with the normalized perpendicular direction.
Returns `true` if the basis was expanded, `false` otherwise.

See also: [`SubspaceBasis`](@ref), [`project`](@ref).
"""
function maybe_expand!(
    sb::SubspaceBasis,
    y::AbstractVector{<:Real},
    config::OracleConfig=OracleConfig();
    tol::Union{Nothing,Float64}=nothing,
)
    tol_eff = something(tol, config.quantization_eps)
    yv = Vector{Float64}(y)
    z = sb.B' * yv
    perp = yv - sb.B * z
    nperp = norm(perp)
    if nperp > tol_eff
        sb.B = hcat(sb.B, perp ./ nperp)
        return true
    end
    return false
end

"""
    cy_normalize(z, c, config=OracleConfig(); tol=nothing) -> Vector{Float64}

Normalize vector `z` so that `c ⋅ z = 1`. Errors if `|c ⋅ z| ≤ tol` (default:
`config.cy_normalize_tol`), which indicates the vector lies in the hyperplane `{z : c'z = 0}`.

This is the standard normalization used in the separation oracle to place rays on the
normalization slice `{y : c'y = 1}`.
"""
function cy_normalize(
    z::AbstractVector{<:Real},
    c::AbstractVector{<:Real},
    config::OracleConfig=OracleConfig();
    tol::Union{Nothing,Float64}=nothing,
)
    tol_eff = something(tol, config.cy_normalize_tol)
    w = Vector{Float64}(z)
    s = dot(Vector{Float64}(c), w)
    abs(s) <= tol_eff && error("cy_normalize: dot(c, z) is ~0; cannot normalize")
    return w ./ s
end
