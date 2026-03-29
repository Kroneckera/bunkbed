function _rationalize_basis_matrix(B::AbstractMatrix{<:Real}, config::OracleConfig)
    size(B, 2) == 0 && return zeros(BigInt, size(B, 1), 0)
    out = Matrix{BigInt}(undef, size(B, 1), size(B, 2))
    for col in axes(B, 2)
        out[:, col] = rationalize_ray_coordwise(view(B, :, col), config)
    end
    return out
end

"""
    save_checkpoint(file, R, iter, lp_calls, B=nothing, config=OracleConfig())

Serialize the current oracle state to `file` for resumable computation. Rays in `R` and
the optional subspace basis `B` are rationalized to `BigInt` before serialization to avoid
floating-point drift across sessions.

See also: [`load_checkpoint`](@ref), [`OracleConfig`](@ref).
"""
function save_checkpoint(
    file::String,
    R::Vector{Vector{Float64}},
    iter::Int,
    lp_calls::Int,
    B::Union{Matrix{Float64},Nothing}=nothing,
    config::OracleConfig=OracleConfig(),
)
    dir = dirname(file)
    !isempty(dir) && mkpath(dir)
    payload = CheckpointPayload(
        iter,
        lp_calls,
        [rationalize_ray_coordwise(ray, config) for ray in R],
        B === nothing ? nothing : _rationalize_basis_matrix(B, config),
    )
    open(file, "w") do io
        serialize(io, payload)
    end
    return nothing
end

"""
    load_checkpoint(file, config=OracleConfig()) -> (R, seen_rays, iter, lp_calls, B)

Deserialize a checkpoint previously saved by [`save_checkpoint`](@ref).

Returns a tuple of:
- `R::Vector{Vector{Float64}}` — restored rays.
- `seen_rays::Set{HashKey}` — hash keys of restored rays.
- `iter::Int` — iteration count at checkpoint.
- `lp_calls::Int` — LP call count at checkpoint.
- `B::Union{Matrix{Float64}, Nothing}` — restored subspace basis (re-orthogonalized if
  `config.checkpoint_reorthogonalize` is `true`).

See also: [`save_checkpoint`](@ref).
"""
function load_checkpoint(file::String, config::OracleConfig=OracleConfig())
    isfile(file) || throw(ArgumentError("load_checkpoint: checkpoint file not found: $file"))
    payload = try
        open(file, "r") do io
            deserialize(io)
        end
    catch err
        throw(ArgumentError("load_checkpoint: failed to deserialize checkpoint $file: $(sprint(showerror, err))"))
    end

    payload isa CheckpointPayload ||
        throw(ArgumentError("load_checkpoint: invalid checkpoint payload in $file"))

    R = [Float64.(ray) for ray in payload.rays]
    seen_rays = Set{HashKey}(hash_key_ray(ray, config) for ray in R)
    B = if payload.basis === nothing
        nothing
    else
        BF = Float64.(payload.basis)
        config.checkpoint_reorthogonalize ? reorthogonalize_basis(BF) : BF
    end
    return R, seen_rays, payload.iter, payload.lp_calls, B
end
