include(joinpath(@__DIR__, "test_helpers.jl"))

@testset "Paper-family regression" begin
    result = enumerate_paper_family_regression()
    @test result.counts == [25, 240, 240, 60, 700]
    @test result.count == 1265
    @test result.hash == "7e15670f2c853fa80464575213bf190f8deb9e8e"

    artifact_path = joinpath(PKG_ROOT, "data", "valid_partition_tuples_nobs3_notebook.txt")
    if isfile(artifact_path)
        baseline_lines = sort(readlines(artifact_path))
        @test sort(regression_tuple_line.(3, result.tuples)) == baseline_lines
    end
end
