import normaliz_jll

"""
    AbstractNormalizBackend

Abstract type for Normaliz computation backends.
"""
abstract type AbstractNormalizBackend end

"""
    NormalizCLIBackend()

Normaliz backend that invokes the `normaliz` command-line tool via `normaliz_jll`.
Writes `.in` files, runs the binary, and parses `.sup`/`.ext`/`.out` output files.
This is the default and only production backend.
"""
struct NormalizCLIBackend <: AbstractNormalizBackend
end

"""
    NormalizResult

Result of a Normaliz cone computation.

# Fields
- `support_hyperplanes::Matrix{BigInt}` — facet normals (rows).
- `extreme_rays::Matrix{BigInt}` — extreme rays (rows).
- `lineality_space::Matrix{BigInt}` — basis of the lineality space (rows; empty if pointed).
"""
struct NormalizResult
    support_hyperplanes::Matrix{BigInt}
    extreme_rays::Matrix{BigInt}
    lineality_space::Matrix{BigInt}
end

"""
    normaliz_backend_status() -> Symbol

Return the active Normaliz backend type. Currently always `:normaliz_cli`.
"""
normaliz_backend_status() = :normaliz_cli

"""
    is_pointed(result::NormalizResult) -> Bool

Return `true` if the computed cone has an empty lineality space (i.e., is pointed).
A pointed cone is required for facet enumeration in the projection algorithm.

See also: [`compute_cone`](@ref), [`enumerate_facets`](@ref).
"""
is_pointed(result::NormalizResult) = size(result.lineality_space, 1) == 0

_empty_bigint_matrix(ncols::Integer) = Matrix{BigInt}(undef, 0, Int(ncols))

function _matrix_from_rows(rows::Vector{<:AbstractVector{BigInt}}, ncols::Integer)
    isempty(rows) && return _empty_bigint_matrix(ncols)
    out = Matrix{BigInt}(undef, length(rows), Int(ncols))
    for (i, row) in enumerate(rows)
        length(row) == ncols || error("Normaliz parser: expected row length $(ncols), got $(length(row))")
        @inbounds out[i, :] = row
    end
    return out
end

function _parse_bigint_row(line::AbstractString, ncols::Integer)
    parts = split(strip(line))
    length(parts) == ncols || error("Normaliz parser: expected $(ncols) columns, got $(length(parts)) in `$line`")
    return parse.(BigInt, parts)
end

function _try_parse_bigint_row(line::AbstractString)
    stripped = strip(line)
    isempty(stripped) && return nothing
    parts = split(stripped)
    isempty(parts) && return nothing
    try
        return parse.(BigInt, parts)
    catch
        return nothing
    end
end

function _normaliz_flags(config::OracleConfig)
    flags = String["-a", "-B"]
    config.normaliz_verbose && push!(flags, "-c")
    config.normaliz_threads > 0 && push!(flags, "-x=$(config.normaliz_threads)")
    config.normaliz_keep_order && push!(flags, "-k")
    if config.normaliz_algorithm == :projection
        push!(flags, "-j")
    elseif config.normaliz_algorithm == :projection_float
        push!(flags, "-J")
    end
    return flags
end

function write_normaliz_input(path::AbstractString, generators::AbstractMatrix{<:Integer}; input_type::Symbol=:cone)
    nrows, ncols = size(generators)
    input_type in (:cone, :vertices) || throw(ArgumentError("write_normaliz_input: input_type must be :cone or :vertices"))
    tag = input_type === :cone ? "cone" : "vertices"
    open(path, "w") do io
        println(io, "amb_space ", ncols)
        println(io, tag, " ", nrows)
        for i in 1:nrows
            println(io, join(generators[i, :], " "))
        end
        println(io, "SupportHyperplanes")
        println(io, "ExtremeRays")
    end
    return path
end

function _parse_matrix_file(path::AbstractString, ambient_dim::Integer)
    if !isfile(path)
        return nothing
    end
    lines = readlines(path)
    isempty(lines) && return _empty_bigint_matrix(ambient_dim)

    count = parse(Int, strip(lines[1]))
    dim = length(lines) >= 2 ? parse(Int, strip(lines[2])) : Int(ambient_dim)
    count == 0 && return _empty_bigint_matrix(dim)
    length(lines) >= count + 2 || error("Normaliz parser: file $(path) ended before $(count) rows were available")

    rows = Vector{Vector{BigInt}}(undef, count)
    for i in 1:count
        rows[i] = _parse_bigint_row(lines[i + 2], dim)
    end
    return _matrix_from_rows(rows, dim)
end

function _parse_out_section(path::AbstractString, header::Regex, ambient_dim::Integer)
    isfile(path) || return nothing
    lines = readlines(path)
    for (idx, line) in enumerate(lines)
        m = match(header, strip(line))
        m === nothing && continue
        count = parse(Int, m.captures[1])
        count == 0 && return _empty_bigint_matrix(ambient_dim)

        rows = Vector{Vector{BigInt}}()
        j = idx + 1
        while j <= length(lines) && length(rows) < count
            parsed = _try_parse_bigint_row(lines[j])
            if parsed !== nothing
                push!(rows, parsed)
            end
            j += 1
        end
        length(rows) == count || error("Normaliz parser: section $(header) in $(path) expected $(count) rows, found $(length(rows))")
        return _matrix_from_rows(rows, length(rows[1]))
    end
    return nothing
end

function _parse_first_matrix_file(base::AbstractString, ambient_dim::Integer, suffixes)
    for suffix in suffixes
        parsed = _parse_matrix_file(base * suffix, ambient_dim)
        parsed !== nothing && return parsed
    end
    return nothing
end

function _parse_support_hyperplanes(base::AbstractString, ambient_dim::Integer)
    parsed = _parse_first_matrix_file(base, ambient_dim, (".cst", ".sup", ".esp"))
    parsed !== nothing && return parsed
    parsed = _parse_out_section(base * ".out", r"^(\d+) support hyperplanes:$", ambient_dim)
    return something(parsed, _empty_bigint_matrix(ambient_dim))
end

function _parse_extreme_rays(base::AbstractString, ambient_dim::Integer)
    parsed = _parse_matrix_file(base * ".ext", ambient_dim)
    parsed !== nothing && return parsed
    parsed = _parse_out_section(base * ".out", r"^(\d+) extreme rays:$", ambient_dim)
    return something(parsed, _empty_bigint_matrix(ambient_dim))
end

function _parse_lineality_space(base::AbstractString, ambient_dim::Integer)
    parsed = _parse_first_matrix_file(base, ambient_dim, (".msp",))
    parsed !== nothing && return parsed
    parsed = _parse_out_section(base * ".out", r"^(\d+) basis elements of maximal subspace:$", ambient_dim)
    return something(parsed, _empty_bigint_matrix(ambient_dim))
end

function _run_normaliz(base::AbstractString, config::OracleConfig)
    flags = _normaliz_flags(config)
    stdout_target = config.normaliz_verbose ? config.log_io : devnull
    stderr_target = config.log_io
    base_cmd = normaliz_jll.normaliz()
    cmd = Cmd(Cmd(vcat(base_cmd.exec, flags, [base])); env=base_cmd.env, dir=base_cmd.dir)
    run(pipeline(cmd; stdout=stdout_target, stderr=stderr_target))
    return nothing
end

"""
    compute_cone(backend, generators, config=OracleConfig(); input_type=:cone) -> NormalizResult

Compute support hyperplanes, extreme rays, and lineality space of the cone generated by
the rows of `generators` using Normaliz.

# Arguments
- `backend::AbstractNormalizBackend` — Normaliz backend (e.g., `NormalizCLIBackend()`).
- `generators::AbstractMatrix{<:Integer}` — generator matrix (rows are generators).
- `config::OracleConfig` — controls Normaliz flags, threads, verbosity.
- `input_type::Symbol` — `:cone` (default) for cone generators, `:vertices` for vertex input.

Returns a [`NormalizResult`](@ref) with `support_hyperplanes`, `extreme_rays`, and
`lineality_space` fields.

See also: [`enumerate_facets`](@ref), [`is_pointed`](@ref), [`NormalizResult`](@ref).
"""
function compute_cone(
    ::NormalizCLIBackend,
    generators::AbstractMatrix{<:Integer},
    config::OracleConfig=OracleConfig();
    input_type::Symbol=:cone,
)
    ambient_dim = size(generators, 2)
    ambient_dim > 0 || throw(ArgumentError("compute_cone: ambient dimension must be positive"))
    if size(generators, 1) == 0
        empty = _empty_bigint_matrix(ambient_dim)
        return NormalizResult(empty, empty, empty)
    end

    G = Matrix{BigInt}(generators)
    tmpdir = mktempdir()
    base = joinpath(tmpdir, "cone")
    try
        write_normaliz_input(base * ".in", G; input_type=input_type)
        _run_normaliz(base, config)
        support_hyperplanes = _parse_support_hyperplanes(base, ambient_dim)
        extreme_rays = _parse_extreme_rays(base, ambient_dim)
        lineality_space = _parse_lineality_space(base, ambient_dim)
        return NormalizResult(support_hyperplanes, extreme_rays, lineality_space)
    finally
        rm(tmpdir; recursive=true, force=true)
    end
end
