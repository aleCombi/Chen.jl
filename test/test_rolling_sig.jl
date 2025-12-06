using Test
using ChenSignatures

@testset "Rolling Window Signatures" begin
    @testset "Basic functionality" begin
        # Test case: 10 points, window_size=5, stride=1 (default)
        path = randn(10, 2)
        m = 3
        window_size = 5

        result = rolling_sig(path, m, window_size)

        # Expected: (10 - 5) / 1 + 1 = 6 windows
        # Signature dim: 2 + 4 + 8 = 14
        @test size(result) == (14, 6)

        # Verify first window matches direct computation
        window1 = path[1:5, :]
        sig1 = sig(window1, m)
        @test result[:, 1] ≈ sig1

        # Verify last window
        window6 = path[6:10, :]
        sig6 = sig(window6, m)
        @test result[:, 6] ≈ sig6
    end

    @testset "Different strides" begin
        path = randn(20, 3)
        m = 2
        window_size = 5

        # Stride = 1 (default, maximum overlap)
        result_s1 = rolling_sig(path, m, window_size; stride=1)
        @test size(result_s1, 2) == 16  # (20 - 5) / 1 + 1

        # Stride = 2
        result_s2 = rolling_sig(path, m, window_size; stride=2)
        @test size(result_s2, 2) == 8   # (20 - 5) / 2 + 1

        # Stride = 5 (non-overlapping)
        result_s5 = rolling_sig(path, m, window_size; stride=5)
        @test size(result_s5, 2) == 4   # (20 - 5) / 5 + 1

        # Verify specific windows match
        @test result_s2[:, 1] ≈ result_s1[:, 1]  # First window same
        @test result_s2[:, 2] ≈ result_s1[:, 3]  # Second window at index 3
    end

    @testset "Edge cases" begin
        # Minimum path length (window_size = N)
        path_min = randn(5, 2)
        result_min = rolling_sig(path_min, 2, 5)
        @test size(result_min, 2) == 1  # Only one window
        @test result_min[:, 1] ≈ sig(path_min, 2)

        # Single dimension
        path_1d = randn(10, 1)
        result_1d = rolling_sig(path_1d, 3, 4)
        @test size(result_1d) == (3, 7)  # dim: 1+1+1=3, windows: 7

        # High dimension
        path_hd = randn(15, 4)
        result_hd = rolling_sig(path_hd, 2, 6; stride=3)
        @test size(result_hd) == (20, 4)  # dim: 4+16=20, windows: 4
    end

    @testset "Input validation" begin
        path = randn(10, 2)

        # Window size too small
        @test_throws ArgumentError rolling_sig(path, 3, 1)

        # Window size exceeds path length
        @test_throws ArgumentError rolling_sig(path, 3, 15)

        # Invalid stride
        @test_throws ArgumentError rolling_sig(path, 3, 5; stride=0)
        @test_throws ArgumentError rolling_sig(path, 3, 5; stride=-1)

        # Invalid signature level
        @test_throws ArgumentError rolling_sig(path, 0, 5)

        # Path too short
        @test_throws ArgumentError rolling_sig(randn(1, 2), 3, 5)
    end

    @testset "Type stability and consistency" begin
        # Float64
        path_f64 = randn(Float64, 12, 2)
        result_f64 = rolling_sig(path_f64, 3, 4)
        @test eltype(result_f64) == Float64

        # Float32
        path_f32 = randn(Float32, 12, 2)
        result_f32 = rolling_sig(path_f32, 3, 4)
        @test eltype(result_f32) == Float32
    end

    @testset "Correctness: all windows" begin
        # Verify every window matches individual computation
        path = randn(15, 2)
        m = 3
        window_size = 6
        stride = 2

        result = rolling_sig(path, m, window_size; stride=stride)
        num_windows = size(result, 2)

        for i in 1:num_windows
            start_idx = 1 + (i - 1) * stride
            end_idx = start_idx + window_size - 1
            window = path[start_idx:end_idx, :]
            expected_sig = sig(window, m)
            @test result[:, i] ≈ expected_sig atol=1e-12
        end
    end

    @testset "Stride larger than window_size" begin
        # Test with stride > window_size (gaps between windows)
        path = randn(30, 2)
        m = 2
        window_size = 5
        stride = 10

        result = rolling_sig(path, m, window_size; stride=stride)
        @test size(result, 2) == 3  # (30 - 5) / 10 + 1 = 3

        # Verify each window
        @test result[:, 1] ≈ sig(path[1:5, :], m)
        @test result[:, 2] ≈ sig(path[11:15, :], m)
        @test result[:, 3] ≈ sig(path[21:25, :], m)
    end
end
