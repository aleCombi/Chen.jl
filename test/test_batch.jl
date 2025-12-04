using Test
using ChenSignatures

@testset "Batch Processing" begin
    @testset "sig with 3D arrays" begin
        # Test basic functionality
        paths = randn(10, 2, 5)
        sigs = sig(paths, 3; threaded=false)

        @test size(sigs) == (14, 5)  # 2 + 4 + 8 = 14 signature length

        # Verify each column matches single path computation
        for i in 1:5
            sig_single = sig(paths[:, :, i], 3)
            @test sig_single ≈ sigs[:, i]
        end

        # Test threading produces same results
        sigs_threaded = sig(paths, 3; threaded=true)
        @test sigs ≈ sigs_threaded
    end

    @testset "logsig with 3D arrays" begin
        # Test basic functionality
        paths = randn(10, 2, 5)
        basis = prepare(2, 3)
        logsigs = logsig(paths, basis; threaded=false)

        @test size(logsigs, 2) == 5  # 5 paths

        # Verify each column matches single path computation
        for i in 1:5
            logsig_single = logsig(paths[:, :, i], basis)
            @test logsig_single ≈ logsigs[:, i]
        end

        # Test threading produces same results
        logsigs_threaded = logsig(paths, basis; threaded=true)
        @test logsigs ≈ logsigs_threaded
    end

    @testset "Different dimensions and levels" begin
        # 3D paths, level 4
        paths_3d = randn(20, 3, 10)
        sigs_3d = sig(paths_3d, 4)
        @test size(sigs_3d) == (120, 10)  # 3 + 9 + 27 + 81 = 120

        # 4D paths, level 2
        paths_4d = randn(15, 4, 8)
        sigs_4d = sig(paths_4d, 2)
        @test size(sigs_4d) == (20, 8)  # 4 + 16 = 20

        # Verify correctness for first path
        @test sig(paths_3d[:, :, 1], 4) ≈ sigs_3d[:, 1]
        @test sig(paths_4d[:, :, 1], 2) ≈ sigs_4d[:, 1]
    end

    @testset "Single batch path" begin
        # Edge case: batch of size 1
        path_single = randn(10, 2, 1)
        sig_batch = sig(path_single, 3)
        sig_regular = sig(path_single[:, :, 1], 3)

        @test size(sig_batch) == (14, 1)
        @test sig_batch[:, 1] ≈ sig_regular
    end

    @testset "Input validation" begin
        # Too few points
        @test_throws ArgumentError sig(randn(1, 2, 5), 3)

        # Invalid dimension
        @test_throws ArgumentError sig(randn(10, 0, 5), 3)

        # Invalid level
        @test_throws ArgumentError sig(randn(10, 2, 5), 0)

        # Zero batch size
        @test_throws ArgumentError sig(randn(10, 2, 0), 3)
    end

    @testset "Type stability" begin
        # Float64
        paths_f64 = randn(Float64, 10, 2, 5)
        sigs_f64 = sig(paths_f64, 3)
        @test eltype(sigs_f64) == Float64

        # Float32
        paths_f32 = randn(Float32, 10, 2, 5)
        sigs_f32 = sig(paths_f32, 3)
        @test eltype(sigs_f32) == Float32
    end

    @testset "logsig dimension mismatch" begin
        paths = randn(10, 3, 5)
        basis_wrong = prepare(2, 3)  # Wrong dimension

        @test_throws ArgumentError logsig(paths, basis_wrong)
    end
end
