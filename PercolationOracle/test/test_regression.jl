include(joinpath(@__DIR__, "test_helpers.jl"))

@testset "Paper-family regression" begin
    result = enumerate_paper_family_regression()
    @test result.counts == [25, 240, 240, 60, 700]
    @test result.count == 1265
    @test result.hash == "a7b38fd420c769e00b80659729b95c11b58dec3a"

    raw_tuples = load_feasible_tuples_raw()
    normalized_tuples = load_feasible_tuples()
    normalized_flat = [PartitionID[pair[k] for pair in tuple_pairs for k in 1:2] for tuple_pairs in normalized_tuples]
    @test length(raw_tuples) == length(normalized_tuples) == result.count
    @test archive_tuple_to_paper_pairs(first(raw_tuples), 3, 4) == first(normalized_tuples)
    @test Set(Tuple.(normalized_flat)) == Set(Tuple.(result.tuples))

    artifact_path = joinpath(PKG_ROOT, "data", "valid_partition_tuples_nobs3_notebook.txt")
    if isfile(artifact_path)
        baseline_lines = sort(readlines(artifact_path))
        raw_zero_based = [Int.(tuple) .- 1 for tuple in raw_tuples]
        @test sort(regression_tuple_line.(3, raw_zero_based)) == baseline_lines
    end
end
