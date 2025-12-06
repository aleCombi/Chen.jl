# GPU Tensor Tests
# Tests that Tensor works with GPU arrays (CuArray)

using CUDA
using ChenSignatures
using StaticArrays
using Test

@testset "GPU Tensor Basic Operations" begin
    if !CUDA.functional()
        @warn "CUDA not available, skipping GPU tests"
        return
    end

    # Test parameters
    D, M = 2, 4
    T = Float32

    @testset "Tensor construction with CuArray" begin
        # Create tensor on CPU
        tensor_cpu = ChenSignatures.Tensor{T, D, M}()

        # Check it's a Vector
        @test tensor_cpu.coeffs isa Vector{T}

        # Create tensor with CuArray
        offsets_cpu = ChenSignatures.level_starts0(D, M)
        coeffs_gpu = CUDA.zeros(T, offsets_cpu[end])
        tensor_gpu = ChenSignatures.Tensor{T, D, M}(coeffs_gpu)

        # Check it's a CuArray
        @test tensor_gpu.coeffs isa CuArray{T}
        @test length(tensor_gpu.coeffs) == length(tensor_cpu.coeffs)
    end

    @testset "Horner update on GPU" begin
        # Create tensors
        tensor_cpu = ChenSignatures.Tensor{T, D, M}()
        ChenSignatures._reset_tensor!(tensor_cpu)

        # Create GPU version
        offsets_cpu = ChenSignatures.level_starts0(D, M)
        coeffs_gpu = CuArray(tensor_cpu.coeffs)  # Copy to GPU
        tensor_gpu = ChenSignatures.Tensor{T, D, M}(coeffs_gpu)

        # Create workspace
        ws_len = D^(M-1)
        B1_cpu = zeros(T, ws_len)
        B2_cpu = zeros(T, ws_len)
        B1_gpu = CUDA.zeros(T, ws_len)
        B2_gpu = CUDA.zeros(T, ws_len)

        # Test increment
        z = SVector{D, T}(1.0f0, 2.0f0)

        # Update on CPU
        ChenSignatures.update_signature_horner!(tensor_cpu, z, B1_cpu, B2_cpu)

        # Update on GPU (will use scalar indexing, so wrap in allowscalar)
        CUDA.@allowscalar begin
            ChenSignatures.update_signature_horner!(tensor_gpu, z, B1_gpu, B2_gpu)
        end

        # Compare results
        coeffs_from_gpu = Array(tensor_gpu.coeffs)
        @test coeffs_from_gpu ≈ tensor_cpu.coeffs rtol=1e-6

        println("  ✓ GPU Horner update matches CPU")
    end

    @testset "Multiple Horner updates on GPU" begin
        # Create tensors
        tensor_cpu = ChenSignatures.Tensor{T, D, M}()
        ChenSignatures._reset_tensor!(tensor_cpu)

        offsets_cpu = ChenSignatures.level_starts0(D, M)
        coeffs_gpu = CuArray(tensor_cpu.coeffs)
        tensor_gpu = ChenSignatures.Tensor{T, D, M}(coeffs_gpu)

        # Workspace
        ws_len = D^(M-1)
        B1_cpu = zeros(T, ws_len)
        B2_cpu = zeros(T, ws_len)
        B1_gpu = CUDA.zeros(T, ws_len)
        B2_gpu = CUDA.zeros(T, ws_len)

        # Multiple increments
        increments = [
            SVector{D, T}(1.0f0, 0.0f0),
            SVector{D, T}(0.5f0, 0.5f0),
            SVector{D, T}(0.0f0, 1.0f0)
        ]

        for z in increments
            # CPU
            ChenSignatures.update_signature_horner!(tensor_cpu, z, B1_cpu, B2_cpu)

            # GPU
            CUDA.@allowscalar begin
                ChenSignatures.update_signature_horner!(tensor_gpu, z, B1_gpu, B2_gpu)
            end
        end

        # Compare
        coeffs_from_gpu = Array(tensor_gpu.coeffs)
        @test coeffs_from_gpu ≈ tensor_cpu.coeffs rtol=1e-5

        println("  ✓ Multiple GPU Horner updates match CPU")
    end

    @testset "Tensor copy operations with GPU" begin
        tensor_cpu = ChenSignatures.Tensor{T, D, M}()
        ChenSignatures._reset_tensor!(tensor_cpu)

        # Initialize with some values
        ws_len = D^(M-1)
        B1 = zeros(T, ws_len)
        B2 = zeros(T, ws_len)
        z = SVector{D, T}(0.5f0, 1.5f0)
        ChenSignatures.update_signature_horner!(tensor_cpu, z, B1, B2)

        # Copy to GPU
        tensor_gpu = ChenSignatures.Tensor{T, D, M}(CuArray(tensor_cpu.coeffs))

        # Test similar
        tensor_gpu_similar = similar(tensor_gpu)
        @test tensor_gpu_similar.coeffs isa CuArray{T}
        @test length(tensor_gpu_similar.coeffs) == length(tensor_gpu.coeffs)

        # Test copy
        tensor_gpu_copy = copy(tensor_gpu)
        @test tensor_gpu_copy.coeffs isa CuArray{T}
        @test Array(tensor_gpu_copy.coeffs) ≈ tensor_cpu.coeffs

        println("  ✓ Tensor copy operations work with GPU arrays")
    end
end

@testset "GPU signature_path! Integration" begin
    if !CUDA.functional()
        @warn "CUDA not available, skipping GPU signature_path! tests"
        return
    end

    # Test parameters
    D, M = 2, 4
    T = Float32
    N = 10  # Path length

    @testset "signature_path! with Matrix input on GPU" begin
        # Create a simple path on CPU
        path_cpu = Float32[
            0.0 0.0;
            1.0 0.0;
            1.0 1.0;
            2.0 1.0;
            2.0 2.0;
            3.0 2.0;
            3.0 3.0;
            4.0 3.0;
            4.0 4.0;
            5.0 5.0
        ]

        # Compute signature on CPU (reference)
        tensor_cpu = ChenSignatures.Tensor{T, D, M}()
        ws_cpu = ChenSignatures.SignatureWorkspace{T, D, M}()
        ChenSignatures.signature_path!(tensor_cpu, path_cpu, ws_cpu)

        # Create GPU tensor and workspace
        coeffs_gpu = CuArray(zeros(T, length(tensor_cpu.coeffs)))
        tensor_gpu = ChenSignatures.Tensor{T, D, M}(coeffs_gpu)

        ws_len = D^(M-1)
        B1_gpu = CUDA.zeros(T, ws_len)
        B2_gpu = CUDA.zeros(T, ws_len)
        ws_gpu = ChenSignatures.SignatureWorkspace{T, D, M}(B1_gpu, B2_gpu)

        # Compute signature on GPU
        CUDA.@allowscalar begin
            ChenSignatures.signature_path!(tensor_gpu, path_cpu, ws_gpu)
        end

        # Compare results
        coeffs_from_gpu = Array(tensor_gpu.coeffs)
        @test coeffs_from_gpu ≈ tensor_cpu.coeffs rtol=1e-5

        println("  ✓ signature_path! with Matrix input matches CPU")
    end

    @testset "signature_path! with SVector path on GPU" begin
        # Create path as vector of SVectors
        path_svec_cpu = [
            SVector{D, T}(0.0f0, 0.0f0),
            SVector{D, T}(1.0f0, 0.0f0),
            SVector{D, T}(1.0f0, 1.0f0),
            SVector{D, T}(2.0f0, 1.0f0),
            SVector{D, T}(2.0f0, 2.0f0),
            SVector{D, T}(3.0f0, 2.0f0),
            SVector{D, T}(3.0f0, 3.0f0),
            SVector{D, T}(4.0f0, 3.0f0),
            SVector{D, T}(4.0f0, 4.0f0),
            SVector{D, T}(5.0f0, 5.0f0)
        ]

        # CPU reference
        tensor_cpu = ChenSignatures.Tensor{T, D, M}()
        ws_cpu = ChenSignatures.SignatureWorkspace{T, D, M}()
        ChenSignatures.signature_path!(tensor_cpu, path_svec_cpu, ws_cpu)

        # GPU version
        coeffs_gpu = CuArray(zeros(T, length(tensor_cpu.coeffs)))
        tensor_gpu = ChenSignatures.Tensor{T, D, M}(coeffs_gpu)

        ws_len = D^(M-1)
        B1_gpu = CUDA.zeros(T, ws_len)
        B2_gpu = CUDA.zeros(T, ws_len)
        ws_gpu = ChenSignatures.SignatureWorkspace{T, D, M}(B1_gpu, B2_gpu)

        # Compute on GPU
        CUDA.@allowscalar begin
            ChenSignatures.signature_path!(tensor_gpu, path_svec_cpu, ws_gpu)
        end

        # Compare
        coeffs_from_gpu = Array(tensor_gpu.coeffs)
        @test coeffs_from_gpu ≈ tensor_cpu.coeffs rtol=1e-5

        println("  ✓ signature_path! with SVector input matches CPU")
    end

    @testset "signature_path! without workspace (allocating version)" begin
        # Test the convenience version that auto-creates workspace
        path_cpu = Float32[
            0.0 0.0;
            1.0 1.0;
            2.0 1.0;
            3.0 2.0
        ]

        # CPU reference
        tensor_cpu = ChenSignatures.Tensor{T, D, M}()
        ChenSignatures.signature_path!(tensor_cpu, path_cpu)

        # GPU version with GPU tensor
        coeffs_gpu = CuArray(zeros(T, length(tensor_cpu.coeffs)))
        tensor_gpu = ChenSignatures.Tensor{T, D, M}(coeffs_gpu)

        # This will create CPU workspace internally, but that's OK for validation
        CUDA.@allowscalar begin
            ChenSignatures.signature_path!(tensor_gpu, path_cpu)
        end

        # Compare
        coeffs_from_gpu = Array(tensor_gpu.coeffs)
        @test coeffs_from_gpu ≈ tensor_cpu.coeffs rtol=1e-5

        println("  ✓ signature_path! without workspace works with GPU tensor")
    end
end

println("\nGPU Tensor and signature_path! tests completed successfully!")
