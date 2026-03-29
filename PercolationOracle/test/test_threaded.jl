include(joinpath(@__DIR__, "test_helpers.jl"))

@testset "Threaded enumeration" begin
    @testset "Paper-family regression (n_threads=$nt)" for nt in [1, 2, 4]
        result = enumerate_paper_family_regression_threaded(n_threads=nt)
        @test result.counts == [25, 240, 240, 60, 700]
        @test result.count == 1265
        @test result.hash == "7e15670f2c853fa80464575213bf190f8deb9e8e"
    end

    @testset "Matches single-threaded exactly" begin
        ref = enumerate_paper_family_regression()
        for nt in [2, 4]
            threaded = enumerate_paper_family_regression_threaded(n_threads=nt)
            @test threaded.count == ref.count
            @test threaded.hash == ref.hash
            @test threaded.counts == ref.counts
        end
    end

    @testset "Small instance: single condensation" begin
        n_obs = 3
        trees = paper_family_decision_trees()
        # Use condensation 0 (no edges = 1|2|3 partition)
        condensation = condensation_from_partition_id(n_obs, PartitionID(4))
        colorings = PackedColoring[apply_decision_tree_packed(n_obs, condensation, tree) for tree in trees[2:end]]
        ctx = PackedOracleContext(n_obs, condensation, colorings)

        ref = enumerate_compatible_partitions_packed(ctx)
        for nt in [1, 2, 4]
            threaded = enumerate_compatible_partitions_threaded(ctx, nt)
            @test length(threaded) == length(ref)
            @test sort(threaded) == sort(ref)
        end
    end

    @testset "Prefix passthrough" begin
        n_obs = 3
        trees = paper_family_decision_trees()
        condensation = condensation_from_partition_id(n_obs, PartitionID(0))
        colorings = PackedColoring[apply_decision_tree_packed(n_obs, condensation, tree) for tree in trees[2:end]]
        ctx = PackedOracleContext(n_obs, condensation, colorings)

        # With a fixed prefix, threaded should match single-threaded
        prefix = [PartitionID(0)]
        ref = enumerate_compatible_partitions_packed(ctx, prefix)
        for nt in [2, 4]
            threaded = enumerate_compatible_partitions_threaded(ctx, nt; prefix=prefix)
            @test length(threaded) == length(ref)
            @test sort(threaded) == sort(ref)
        end
    end
end
