using Test
using ChenSignatures
using StaticArrays

@testset "time_augment matrix" begin
    path = Float32[1 2; 3 4; 5 6]
    out = time_augment(path; Tspan=2f0)
    @test size(out) == (3, 3)
    @test eltype(out) == Float32
    @test out[:, 1] == Float32[0, 1, 2]
    @test out[:, 2:end] == path

    custom_times = [0.0, 0.1, 0.4]
    out2 = time_augment(Float64.(path); times=custom_times)
    @test out2[:, 1] == custom_times
    @test out2[:, 2:end] == Float64.(path)
end

@testset "time_augment SVector and batch" begin
    svec_path = [SVector{2,Float64}(1.0, 2.0), SVector{2,Float64}(3.0, 4.0)]
    out = time_augment(svec_path; times=[0.0, 0.5])
    @test length(out) == 2
    @test out[1] == SVector{3,Float64}(0.0, 1.0, 2.0)
    @test out[2] == SVector{3,Float64}(0.5, 3.0, 4.0)

    paths = reshape(Float32[1, 2, 3, 4, 5, 6], 3, 2, 1)
    out_batch = time_augment(paths; Tspan=1f0)
    @test size(out_batch) == (3, 3, 1)
    @test out_batch[:, 1, 1] == Float32[0, 0.5, 1.0]
    @test out_batch[:, 2:end, 1] == paths[:, :, 1]
end

@testset "lead_lag transforms" begin
    path = reshape(Float64[0, 1, 2], 3, 1)
    expected = [0 0; 0 1; 1 1; 1 2; 2 2]
    out = lead_lag(path)
    @test size(out) == (5, 2)
    @test out == expected

    svec_path = [SVector{2,Float64}(1.0, 2.0), SVector{2,Float64}(3.0, 4.0)]
    out_svec = lead_lag(svec_path)
    @test length(out_svec) == 3
    @test out_svec[1] == SVector{4,Float64}(1.0, 2.0, 1.0, 2.0)
    @test out_svec[2] == SVector{4,Float64}(1.0, 2.0, 3.0, 4.0)
    @test out_svec[3] == SVector{4,Float64}(3.0, 4.0, 3.0, 4.0)

    batch_paths = Array{Float32}(undef, 2, 2, 1)
    batch_paths[:, :, 1] = [0 1; 2 3]
    out_batch = lead_lag(batch_paths)
    @test size(out_batch) == (3, 4, 1)
    @test out_batch[:, 1:2, 1] == Float32[0 1; 0 1; 2 3]
    @test out_batch[:, 3:4, 1] == Float32[0 1; 2 3; 2 3]
end

@testset "sig/logsig wrappers" begin
    path = reshape(Float64[0, 1, 2], 3, 1)
    m = 2

    @test sig_time(path, m) == sig(time_augment(path), m)
    @test sig_leadlag(path, m) == sig(lead_lag(path), m)

    basis_time = prepare(2, m)  # D=1 -> time augment to 2
    @test logsig_time(path, basis_time) == logsig(time_augment(path), basis_time)
    @test_throws ArgumentError logsig_time(path, prepare(1, m))

    basis_leadlag = prepare(2, m)  # D=1 -> lead-lag to 2
    @test logsig_leadlag(path, basis_leadlag) == logsig(lead_lag(path), basis_leadlag)
end
