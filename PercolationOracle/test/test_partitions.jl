include(joinpath(@__DIR__, "test_helpers.jl"))
using Combinatorics

@testset "PartitionID tables and roundtrips" begin
    for n in 1:MAX_OBSERVABLES
        raw = collect(partitions(1:n))
        @test bell_number(n) == length(raw)
        @test all_partition_ids(n) == PartitionID.(0:bell_number(n)-1)
        for (idx, partition) in enumerate(raw)
            pid = encode_partition(n, partition)
            @test pid == PartitionID(idx - 1)
            decoded = decode_partition(n, pid)
            @test canonical_partition_label(decoded) == canonical_partition_label(partition)
            @test encode_partition(n, decoded) == pid
        end
    end

    @test [partition_label(3, pid) for pid in all_partition_ids(3)] == ["123", "12|3", "13|2", "1|23", "1|2|3"]
    @test partition_block_count(4, encode_partition(4, [[1, 2], [3, 4]])) == 2
    @test collect(partition_assignments(3, encode_partition(3, [[1, 3], [2]]))) == UInt8[1, 2, 1]
end
