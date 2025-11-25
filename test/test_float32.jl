# test/test_float32.jl

using Test
using Chen
using StaticArrays

@testset "Float32 Correctness" begin
    
    @testset "Linear path (d=2, m=3)" begin
        N = 20
        ts64 = range(0.0, stop=1.0, length=N)
        ts32 = range(0.0f0, stop=1.0f0, length=N)
        
        path64 = [SVector(t, 2t) for t in ts64]
        path32 = [SVector(t, 2t) for t in ts32]
        
        sig64 = signature_path(Chen.Tensor{Float64}, path64, 3)
        sig32 = signature_path(Chen.Tensor{Float32}, path32, 3)
        
        # Extract only the actual signature coefficients (skip padding)
        d, m = 2, 3
        coeffs64 = Float64[]
        coeffs32_as64 = Float64[]
        
        for k in 1:m
            start_idx = sig64.offsets[k+1] + 1
            len = d^k
            append!(coeffs64, view(sig64.coeffs, start_idx:start_idx+len-1))
            append!(coeffs32_as64, Float64.(view(sig32.coeffs, start_idx:start_idx+len-1)))
        end
        
        @test isapprox(coeffs64, coeffs32_as64, rtol=1e-5)
    end
    
    @testset "Sin path (d=3, m=4)" begin
        N = 15
        ts64 = range(0.0, stop=1.0, length=N)
        ts32 = range(0.0f0, stop=1.0f0, length=N)
        
        ω64 = 2π
        ω32 = Float32(2π)
        
        path64 = [SVector(sin(ω64*t), sin(2ω64*t), sin(3ω64*t)) for t in ts64]
        path32 = [SVector(sin(ω32*t), sin(2ω32*t), sin(3ω32*t)) for t in ts32]
        
        sig64 = signature_path(Chen.Tensor{Float64}, path64, 4)
        sig32 = signature_path(Chen.Tensor{Float32}, path32, 4)
        
        # Extract only actual coefficients
        d, m = 3, 4
        coeffs64 = Float64[]
        coeffs32_as64 = Float64[]
        
        for k in 1:m
            start_idx = sig64.offsets[k+1] + 1
            len = d^k
            append!(coeffs64, view(sig64.coeffs, start_idx:start_idx+len-1))
            append!(coeffs32_as64, Float64.(view(sig32.coeffs, start_idx:start_idx+len-1)))
        end
        
        @test isapprox(coeffs64, coeffs32_as64, rtol=1e-4)
    end
    
    @testset "Type stability" begin
        path32 = [SVector(0.0f0, 0.0f0), SVector(1.0f0, 1.0f0)]
        sig32 = signature_path(Chen.Tensor{Float32}, path32, 2)
        
        @test eltype(sig32.coeffs) == Float32
        @test sig32.coeffs isa Vector{Float32}
    end
end