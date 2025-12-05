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

println("\nGPU Tensor tests completed successfully!")
