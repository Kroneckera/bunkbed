include(joinpath(@__DIR__, "test_helpers.jl"))

@testset "PackedState roundtrips and indexing" begin
    psi = UInt8[1, 0, 3, 2, 1, 0, 2, 3]
    state = pack_state(4, psi)
    @test state isa UInt32
    @test packed_state_bits(length(psi)) == 27
    component, unpacked = unpack_state(state, length(psi))
    @test component == 4
    @test unpacked == psi

    psi16 = UInt8[mod(i, 4) for i in 1:11]
    state16 = pack_state(5, psi16)
    @test state16 isa UInt64
    component16, unpacked16 = unpack_state(state16, length(psi16))
    @test component16 == 5
    @test unpacked16 == psi16

    radices = [3, 4, 2]
    strides = mixed_radix_strides(4, radices)
    @test strides == [4, 12, 48]
    @test state_count(4, radices) == 96
    small_state = pack_state(2, UInt8[1, 2, 0])
    @test state_index(small_state, 4, radices, strides) == 2 + 1 * 4 + 2 * 12 + 0 * 48
end
