using Test
using StaticArrays
using Chen

include("fixtures.jl")

# Generators returning Matrix{Float64} (N x d) required by api.jl
function make_path_linear(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    path = zeros(Float64, N, d)
    for (i, t) in enumerate(ts)
        path[i, 1] = t
        for j in 2:d; path[i, j] = 2t; end
    end
    return path
end

function make_path_sin(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    path = zeros(Float64, N, d)
    ω = 2π
    for (i, t) in enumerate(ts)
        for j in 1:d; path[i, j] = sin(ω * j * t); end
    end
    return path
end

@testset "Correctness against iisignature" begin
    N = Fixtures.N

    @testset "d=2, m=4, Linear" begin
        d, m = 2, 4
        path = make_path_linear(d, N)
        @test sig(path, m) ≈ Fixtures.SIG_D2_M4_LINEAR atol=1e-12
        
        basis = prepare(d, m)
        @test logsig(path, basis) ≈ Fixtures.LOGSIG_D2_M4_LINEAR atol=1e-12
    end

    @testset "d=2, m=4, Sin" begin
        d, m = 2, 4
        path = make_path_sin(d, N)
        @test sig(path, m) ≈ Fixtures.SIG_D2_M4_SIN atol=1e-12
        
        basis = prepare(d, m)
        @test logsig(path, basis) ≈ Fixtures.LOGSIG_D2_M4_SIN atol=1e-12
    end

    @testset "d=3, m=3, Linear" begin
        d, m = 3, 3
        path = make_path_linear(d, N)
        @test sig(path, m) ≈ Fixtures.SIG_D3_M3_LINEAR atol=1e-12
        
        basis = prepare(d, m)
        @test logsig(path, basis) ≈ Fixtures.LOGSIG_D3_M3_LINEAR atol=1e-12
    end
end