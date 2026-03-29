include(joinpath(@__DIR__, "test_helpers.jl"))

@testset "Transition table size and indexing" begin
    condensation = condensation_from_partition_id(3, PartitionID(0))
    colorings = PackedColoring[apply_decision_tree_packed(3, condensation, tree) for tree in paper_family_decision_trees()]
    tt = build_transition_table(3, colorings)
    @test length(tt.next_psi) == 8 * bell_number(3) * 4 * 4 * 4
    @test sizeof(tt.next_psi) == 8 * bell_number(3) * 4 * 4 * 4
    @test sizeof(tt.next_psi) <= 15_000
    @test mask_value(tt, 1, 0, 4) == 0
end

@testset "Packed transition logic matches baseline neighbors" begin
    condensation = condensation_from_partition_id(3, PartitionID(4))
    selected_trees = paper_family_decision_trees()[2:3]
    colorings = PackedColoring[apply_decision_tree_packed(3, condensation, tree) for tree in selected_trees]
    partition_ids = PartitionID[PartitionID(4), PartitionID(2)]
    ctx = PackedOracleContext(3, condensation, colorings)
    ug, matrices = baseline_universal_graph(condensation, colorings, partition_ids; n_obs=3)

    for active_index in 1:2, component in 1:4, psi1 in 0:3, psi2 in 0:3
        state = pack_state(component, UInt8[psi1, psi2])
        @test packed_neighbors(ctx, partition_ids, state, active_index) == baseline_neighbors(ug, matrices, component, UInt8[psi1, psi2], active_index)
    end
end
