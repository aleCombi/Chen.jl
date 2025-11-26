using Test
using Chen
using StaticArrays

# Helper to create Matrix from logic
function matrix_linear(N, d, type=Float64)
    ts = range(type(0.0), stop=type(1.0), length=N)
    path = zeros(type, N, d)
    for (i, t) in enumerate(ts)
        path[i, 1] = t
        for j in 2:d
            path[i, j] = 2t
        end
    end
    return path
end

@testset "Float32 Correctness" begin
    
    @testset "Linear path (d=2, m=3)" begin
        N = 20
        path64 = matrix_linear(N, 2, Float64)
        path32 = matrix_linear(N, 2, Float32)
        
        # Use high-level API
        sig64 = sig(path64, 3)
        sig32 = sig(path32, 3)
        
        # sig32 should be Vector{Float32}
        @test eltype(sig32) == Float32
        
        # Convert to Float64 for comparison
        @test sig64 ≈ Float64.(sig32) rtol=1e-5
    end
    
    @testset "Sin path (d=3, m=4)" begin
        N = 15
        d, m = 3, 4
        
        # Create Float64 Matrix
        path64 = zeros(Float64, N, d)
        ts64 = range(0.0, 1.0, length=N)
        ω = 2π
        for (i, t) in enumerate(ts64)
            for j in 1:d; path64[i,j] = sin(ω*j*t); end
        end

        # Create Float32 Matrix
        path32 = Float32.(path64)
        
        sig64 = sig(path64, 4)
        sig32 = sig(path32, 4)
        
        @test eltype(sig32) == Float32
        @test sig64 ≈ Float64.(sig32) rtol=1e-4
    end
    
    @testset "Type stability in API" begin
        # Simple 2-point path (2x2 Matrix)
        path32 = Float32[0.0 0.0; 1.0 1.0]
        s = sig(path32, 2)
        
        @test s isa Vector{Float32}
        
        basis = prepare(2, 2)
        l = logsig(path32, basis)
        # Result typically inherits type from L matrix (Float64 by default) 
        # unless we specifically make L Float32, but checking it runs is enough
        @test l isa AbstractVector
    end
end