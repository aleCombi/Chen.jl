using Test
using ChenSignatures

@testset "Rolling Window Signatures" begin
    @testset "Basic functionality" begin
        path = randn(10, 2)
        m = 3
        window_size = 5

        result = rolling_sig(path, m, window_size)

        @test size(result) == (14, 6)  # (10 - 5) / 1 + 1 = 6 windows

        # Verify first and last windows
        @test result[:, 1] ≈ sig(path[1:5, :], m)
        @test result[:, 6] ≈ sig(path[6:10, :], m)
    end

    @testset "Different strides" begin
        path = randn(20, 3)
        m = 2
        window_size = 5

        result_s1 = rolling_sig(path, m, window_size; stride=1)
        @test size(result_s1, 2) == 16

        result_s2 = rolling_sig(path, m, window_size; stride=2)
        @test size(result_s2, 2) == 8
        @test result_s2[:, 1] ≈ result_s1[:, 1]
        @test result_s2[:, 2] ≈ result_s1[:, 3]

        result_s5 = rolling_sig(path, m, window_size; stride=5)
        @test size(result_s5, 2) == 4
    end

    @testset "Edge cases" begin
        path_min = randn(5, 2)
        @test size(rolling_sig(path_min, 2, 5), 2) == 1
        @test rolling_sig(path_min, 2, 5)[:, 1] ≈ sig(path_min, 2)

        path_1d = randn(10, 1)
        @test size(rolling_sig(path_1d, 3, 4)) == (3, 7)

        path_hd = randn(15, 4)
        @test size(rolling_sig(path_hd, 2, 6; stride=3)) == (20, 4)
    end

    @testset "Input validation" begin
        path = randn(10, 2)
        @test_throws ArgumentError rolling_sig(path, 3, 1)
        @test_throws ArgumentError rolling_sig(path, 3, 15)
        @test_throws ArgumentError rolling_sig(path, 3, 5; stride=0)
        @test_throws ArgumentError rolling_sig(path, 3, 5; stride=-1)
        @test_throws ArgumentError rolling_sig(path, 0, 5)
        @test_throws ArgumentError rolling_sig(randn(1, 2), 3, 5)
    end

    @testset "Type stability and consistency" begin
        path_f64 = randn(Float64, 12, 2)
        @test eltype(rolling_sig(path_f64, 3, 4)) == Float64

        path_f32 = randn(Float32, 12, 2)
        @test eltype(rolling_sig(path_f32, 3, 4)) == Float32
    end

    @testset "Correctness across windows" begin
        path = randn(15, 2)
        m = 3
        window_size = 6
        stride = 2

        result = rolling_sig(path, m, window_size; stride=stride)
        num_windows = size(result, 2)

        for i in 1:num_windows
            start_idx = 1 + (i - 1) * stride
            end_idx = start_idx + window_size - 1
            expected = sig(path[start_idx:end_idx, :], m)
            @test result[:, i] ≈ expected atol=1e-12
        end
    end
end
