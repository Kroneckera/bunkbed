include(joinpath(@__DIR__, "test_helpers.jl"))

@testset "Packed oracle G1 compatibility" begin
    condensation = condensation_from_partition_id(3, PartitionID(0))
    coloring = PackedColoring[apply_decision_tree_packed(3, condensation, paper_family_decision_trees()[1])]
    ctx = PackedOracleContext(3, condensation, coloring)
    @test !check_g1_compatibility_packed(ctx, PartitionID[PartitionID(4)])
    @test !check_partition_compatibility_packed(ctx, PartitionID[PartitionID(4)])
    @test check_partition_compatibility_packed(ctx, PartitionID[PartitionID(0)])
end

@testset "Packed oracle matches baseline single checks" begin
    trees = paper_family_decision_trees()
    cases = [
        (PartitionID(0), trees[2:end], PartitionID[0, 0, 0, 0, 0, 0, 0]),
        (PartitionID(4), trees[2:end], PartitionID[4, 4, 4, 4, 4, 4, 4]),
        (PartitionID(1), trees[2:4], PartitionID[1, 4, 0]),
        (PartitionID(2), trees[2:5], PartitionID[2, 3, 4, 0]),
    ]

    for (g1pid, selected_trees, partition_ids) in cases
        condensation = condensation_from_partition_id(3, g1pid)
        colorings = PackedColoring[apply_decision_tree_packed(3, condensation, tree) for tree in selected_trees]
        ctx = PackedOracleContext(3, condensation, colorings)
        @test check_partition_compatibility_packed(ctx, partition_ids) == baseline_check(condensation, colorings, partition_ids; n_obs=3)
    end

    Random.seed!(1234)
    for _ in 1:15
        g1pid = rand(all_partition_ids(3))
        m = rand(1:7)
        selected_trees = trees[2:m+1]
        partition_ids = PartitionID[rand(0:bell_number(3)-1) for _ in 1:m]
        condensation = condensation_from_partition_id(3, g1pid)
        colorings = PackedColoring[apply_decision_tree_packed(3, condensation, tree) for tree in selected_trees]
        ctx = PackedOracleContext(3, condensation, colorings)
        @test check_partition_compatibility_packed(ctx, partition_ids) == baseline_check(condensation, colorings, partition_ids; n_obs=3)
    end
end
