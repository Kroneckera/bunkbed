using Test
using PercolationOracle
using ProjectedConeOracle

function inequality_enumeration_regression_solver()
    return HiGHSAdapter()
end

function canonical_signature_tuple(values)
    return Tuple(canonicalize_integer_ray(values; tol=1e-8, normalize_sign=true))
end

function signature_from_coeffs(labels, coeffs::Dict{String, BigInt})
    return Tuple(get(coeffs, label, BigInt(0)) for label in labels)
end

function monomial_label(labels, left::String, right::String)
    if left == right
        return "mu($left)^2"
    end
    forward = "mu($left)*mu($right)"
    return forward in labels ? forward : "mu($right)*mu($left)"
end

function expected_m2_signatures(labels)
    positivity = Set{Tuple{Vararg{BigInt}}}()
    for label in labels
        label == "mu(123)*mu(1|2|3)" && continue
        push!(positivity, signature_from_coeffs(labels, Dict(label => BigInt(1))))
    end
    nontrivial = signature_from_coeffs(labels, Dict(
        "mu(123)*mu(1|2|3)" => BigInt(1),
        monomial_label(labels, "12|3", "13|2") => BigInt(-1),
        monomial_label(labels, "12|3", "1|23") => BigInt(-1),
    ))
    push!(positivity, nontrivial)
    return positivity
end

@testset "Inequality enumeration m=2 gives the fresh 15-ray signature set" begin
    result = enumerate_all_inequalities(3, 2; solver=inequality_enumeration_regression_solver(), verbose=false, max_iterations=500)
    @test result.projection_result.converged
    @test length(result.labels) == 15
    @test length(result.rays) == 15
    signatures = Set(canonical_signature_tuple(ray) for ray in result.rays)
    @test signatures == expected_m2_signatures(result.labels)
    @test all(endswith(item, " >= 0") for item in result.formatted_inequalities)
    @test result.inequalities == result.facet_normals
end

@testset "Inequality enumeration m=4 recovers Eq 12" begin
    result = enumerate_all_inequalities(3, 4; solver=inequality_enumeration_regression_solver(), verbose=false, max_iterations=5000)
    @test result.projection_result.converged
    @test length(result.labels) == 15
    @test length(result.rays) == 17

    coeffs = Dict(
        "mu(123)*mu(1|2|3)" => BigInt(1),
        monomial_label(result.labels, "12|3", "1|23") => BigInt(-1),
        monomial_label(result.labels, "13|2", "1|23") => BigInt(-1),
        monomial_label(result.labels, "12|3", "13|2") => BigInt(-1),
    )
    expected = signature_from_coeffs(result.labels, coeffs)
    signatures = Set(canonical_signature_tuple(ray) for ray in result.rays)
    @test expected in signatures
    @test any(
        occursin("mu(123)*mu(1|2|3)", item) &&
        (occursin("mu(12|3)*mu(1|23)", item) || occursin("mu(1|23)*mu(12|3)", item)) &&
        (occursin("mu(13|2)*mu(1|23)", item) || occursin("mu(1|23)*mu(13|2)", item)) &&
        occursin("mu(12|3)*mu(13|2)", item)
        for item in result.formatted_inequalities
    )
end
