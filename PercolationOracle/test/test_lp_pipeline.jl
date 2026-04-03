using Test
using SparseArrays
using Serialization
using PercolationOracle

const LP_PKG_ROOT = normpath(joinpath(@__DIR__, ".."))
const LP_REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

function flatten_tables(tables, n_obs::Int; phi_order::Symbol=default_partition_order(n_obs))
    if phi_order !== :baseline
        tables = [reorder_pair_matrix(table, n_obs; from_order=phi_order, to_order=:baseline) for table in tables]
    end
    values = Int[]
    for k in 1:length(tables), p in all_partition_ids(n_obs), q in all_partition_ids(n_obs)
        push!(values, tables[k][Int(p) + 1, Int(q) + 1])
    end
    return values
end

function sparse_row_nnz_counts(M::SparseMatrixCSC)
    counts = zeros(Int, size(M, 1))
    rows = rowvals(M)
    @inbounds for col in 1:size(M, 2)
        for ptr in nzrange(M, col)
            counts[rows[ptr]] += 1
        end
    end
    return counts
end

function offdiag_coefficient(poly, p::PartitionID, q::PartitionID)
    return haskey(poly.offdiag, (p, q)) ? poly.offdiag[(p, q)] : poly.offdiag[(q, p)]
end

function expected_poly11()
    diag = Dict(PartitionID(0)=>0, PartitionID(1)=>0, PartitionID(2)=>0, PartitionID(3)=>1, PartitionID(4)=>0)
    off = Dict(
        (PartitionID(0), PartitionID(1)) => 1,
        (PartitionID(0), PartitionID(2)) => 1,
        (PartitionID(0), PartitionID(3)) => 0,
        (PartitionID(0), PartitionID(4)) => -1,
        (PartitionID(1), PartitionID(2)) => 2,
        (PartitionID(1), PartitionID(3)) => 1,
        (PartitionID(1), PartitionID(4)) => 0,
        (PartitionID(2), PartitionID(3)) => 1,
        (PartitionID(2), PartitionID(4)) => 0,
        (PartitionID(3), PartitionID(4)) => 1,
    )
    return diag, off
end

function expected_poly12()
    diag = Dict(pid => 0 for pid in all_partition_ids(3))
    off = Dict(
        (PartitionID(0), PartitionID(1)) => 0,
        (PartitionID(0), PartitionID(2)) => 0,
        (PartitionID(0), PartitionID(3)) => 0,
        (PartitionID(0), PartitionID(4)) => 1,
        (PartitionID(1), PartitionID(2)) => -1,
        (PartitionID(1), PartitionID(3)) => -1,
        (PartitionID(1), PartitionID(4)) => 0,
        (PartitionID(2), PartitionID(3)) => -1,
        (PartitionID(2), PartitionID(4)) => 0,
        (PartitionID(3), PartitionID(4)) => 0,
    )
    return diag, off
end

function expected_poly7()
    diag = Dict(PartitionID(0)=>0, PartitionID(1)=>1, PartitionID(2)=>1, PartitionID(3)=>1, PartitionID(4)=>0)
    off = Dict(
        (PartitionID(0), PartitionID(1)) => 1,
        (PartitionID(0), PartitionID(2)) => 1,
        (PartitionID(0), PartitionID(3)) => 0,
        (PartitionID(0), PartitionID(4)) => -1,
        (PartitionID(1), PartitionID(2)) => 2,
        (PartitionID(1), PartitionID(3)) => 1,
        (PartitionID(1), PartitionID(4)) => 0,
        (PartitionID(2), PartitionID(3)) => 1,
        (PartitionID(2), PartitionID(4)) => 0,
        (PartitionID(3), PartitionID(4)) => 1,
    )
    return diag, off
end

@testset "LP pipeline ordering helpers" begin
    tuple_one_based = [1, 2, 3, 4, 5, 1, 2, 3]
    @test archive_tuple_to_paper_pairs(tuple_one_based, 3, 4) == (
        (PartitionID(0), PartitionID(1)),
        (PartitionID(4), PartitionID(0)),
        (PartitionID(2), PartitionID(3)),
        (PartitionID(1), PartitionID(2)),
    )
    @test partition_order_labels(3) == ["123", "1|2|3", "1|23", "12|3", "13|2"]

    base_matrix = reshape(collect(1:25), 5, 5)
    @test reorder_pair_matrix(reorder_pair_matrix(base_matrix, 3; from_order=:baseline, to_order=:paper), 3; from_order=:paper, to_order=:baseline) == base_matrix
end

@testset "Constraint matrix and certificate helpers" begin
    raw_F = load_feasible_tuples_raw()
    F = paper_family_feasible_tuples()
    @test first(F) == archive_tuple_to_paper_pairs(first(raw_F), 3, 4)
    M = build_constraint_matrix(F, 3, 4)
    @test size(M) == (1265, 100)
    @test nnz(M) == 1265 * 4
    @test count(x -> !iszero(x), M[1, :]) == 4
    @test count(x -> !iszero(x), M[end, :]) == 4

    cert11 = appendixA_certificate_11()
    cert12 = appendixA_certificate_12()
    @test verify_certificate(cert11, F; n_obs=3, m=4).feasible
    @test verify_certificate(cert11, F; n_obs=3, m=4).minimum == 0
    @test verify_certificate(cert12, F; n_obs=3, m=4).feasible
    @test verify_certificate(cert12, F; n_obs=3, m=4).minimum == 0
end

@testset "Quadratic polynomial extraction" begin
    cert11 = appendixA_certificate_11()
    cert12 = appendixA_certificate_12()
    poly11 = extract_quadratic_polynomial(cert11, 3)
    poly12 = extract_quadratic_polynomial(cert12, 3)
    poly7 = extract_quadratic_polynomial(inequality7_proof_potentials(), 3)

    diag11, off11 = expected_poly11()
    diag12, off12 = expected_poly12()
    diag7, off7 = expected_poly7()

    @test poly11.order == PartitionID[0, 4, 3, 1, 2]
    @test poly12.order == PartitionID[0, 4, 3, 1, 2]
    @test poly7.order == PartitionID[0, 4, 3, 1, 2]
    @test poly11.diagonal == diag11
    @test all(offdiag_coefficient(poly11, p, q) == coeff for ((p, q), coeff) in off11)
    @test poly12.diagonal == diag12
    @test all(offdiag_coefficient(poly12, p, q) == coeff for ((p, q), coeff) in off12)
    @test poly7.diagonal == diag7
    @test all(offdiag_coefficient(poly7, p, q) == coeff for ((p, q), coeff) in off7)
end

@testset "find_inequality toy LP" begin
    M_toy = sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2)
    result = find_inequality(M_toy, basis_direction(2, 2), basis_direction(2, 1))
    @test result.termination_status == PercolationOracle.MOI.OPTIMAL
    @test result.phi[1] ≈ 1.0 atol=1e-8
    @test result.phi[2] ≈ 0.0 atol=1e-8
end

@testset "Target recovery LP reproduces Appendix A certificates" begin
    F = paper_family_feasible_tuples()
    M = build_constraint_matrix(F, 3, 4)

    cert11 = appendixA_certificate_11()
    result11 = recover_target_certificate(M, cert11, basis_direction(100, 31); n_obs=3, m=4, feasible_tuples=F, time_limit_sec=5.0)
    target11 = canonicalize_integer_ray(flatten_tables(cert11, 3))
    @test result11.termination_status == PercolationOracle.MOI.OPTIMAL
    @test result11.objective_value ≈ 0.0 atol=1e-8
    @test result11.primitive_phi == target11
    @test Tuple(result11.polynomial_signature) == Tuple(result11.target_signature)
    @test result11.verification.feasible

    cert12 = appendixA_certificate_12()
    result12 = recover_target_certificate(M, cert12, basis_direction(100, 46); n_obs=3, m=4, feasible_tuples=F, time_limit_sec=5.0)
    target12 = canonicalize_integer_ray(flatten_tables(cert12, 3))
    @test result12.termination_status == PercolationOracle.MOI.OPTIMAL
    @test result12.objective_value ≈ 0.0 atol=1e-8
    @test result12.primitive_phi == target12
    @test Tuple(result12.polynomial_signature) == Tuple(result12.target_signature)
    @test result12.verification.feasible
end

@testset "Systematic search pilot and inequality (7) convention mismatch" begin
    F = paper_family_feasible_tuples()
    M = build_constraint_matrix(F, 3, 4)
    pilot = systematic_inequality_search(
        M,
        3,
        4;
        feasible_tuples=F,
        anchors=collect(1:5),
        directions=vcat(collect(1:20), collect(-1:-1:-20)),
        time_limit_sec=2.0,
        total_time_limit_sec=120.0,
    )
    @test pilot.attempted == 200
    @test !pilot.timed_out
    @test pilot.unique >= 1
    @test all(item.verification.feasible for item in pilot.catalog)

    ineq7_tables = inequality7_proof_potentials()
    obstruction = verify_certificate(ineq7_tables, F; n_obs=3, m=4)
    @test !obstruction.feasible
    @test obstruction.minimum == -1

    fixed = verify_certificate(inequality7_proof_potentials(swap_T3_pair=true), F; n_obs=3, m=4)
    @test fixed.feasible
    @test fixed.minimum == 0
end

n4_artifact_path = joinpath(LP_REPO_ROOT, "data", "round2_n4_m1.jls")
if isfile(n4_artifact_path)
@testset "Round 2 n=4 artifacts and LP matrix shapes" begin
    artifact_m1 = deserialize(joinpath(LP_REPO_ROOT, "data", "round2_n4_m1.jls"))
    @test artifact_m1.family == "m1"
    @test artifact_m1.counts == fill(15, 15)
    @test artifact_m1.hash == "07c815fdddad4a35db54de9fc69a9b02000561fc"
    M_m1 = build_constraint_matrix(artifact_m1.tuples, 4, 1)
    @test size(M_m1) == (225, 225)
    @test nnz(M_m1) == 225
    @test all(==(1), sparse_row_nnz_counts(M_m1))

    artifact_S = deserialize(joinpath(LP_REPO_ROOT, "data", "round2_n4_S.jls"))
    @test artifact_S.family == "S"
    @test artifact_S.counts == [225, 64, 225, 100, 160, 225, 100, 325, 225, 225, 450, 225, 300, 450, 975]
    @test artifact_S.hash == "1c932dd5eb13b22d2251f5a3135c2e4db981273f"
    M_S = build_constraint_matrix(artifact_S.tuples, 4, 2)
    @test size(M_S) == (4274, 450)
    @test nnz(M_S) == 4274 * 2
    @test all(==(2), sparse_row_nnz_counts(M_S))

    artifact_M = deserialize(joinpath(LP_REPO_ROOT, "data", "round2_n4_M.jls"))
    @test artifact_M.family == "M"
    @test artifact_M.counts == [225, 960, 960, 100, 1975, 225, 1300, 4875, 1300, 225, 8200, 4875, 8200, 450, 63450]
    @test artifact_M.hash == "85f4100ec397ae012f81f85f0091bd4d2c778bec"
    M_M = build_constraint_matrix(artifact_M.tuples, 4, 3)
    @test size(M_M) == (97320, 675)
    @test nnz(M_M) == 97320 * 3
    @test all(==(3), sparse_row_nnz_counts(M_M))
end
end  # n=4 artifacts conditional
