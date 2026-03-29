include(joinpath(@__DIR__, "test_helpers.jl"))

@testset "Condensation bitfields and packed colorings" begin
    trees = paper_family_decision_trees()
    for pid in all_partition_ids(3)
        condensation = condensation_from_partition_id(3, pid)
        decoded = decode_condensation(3, condensation)
        @test encode_condensation(3, decoded) == condensation

        for tree in trees
            packed = apply_decision_tree_packed(3, condensation, tree)
            baseline = DecisionTreeAlgorithm.apply_decision_tree(decoded, baseline_tree(tree))
            @test unpack_coloring(3, packed) == baseline
            @test pack_coloring(baseline) == packed
        end
    end
end
