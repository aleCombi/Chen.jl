using Test
using StaticArrays
using Chen

# Load hardcoded reference values
include("fixtures.jl")

# --- Path Generators (Must match Python generation logic) ---

function make_path_linear(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    [SVector{d,Float64}(ntuple(i -> (i == 1 ? t : 2t), d)) for t in ts]
end

function make_path_sin(d::Int, N::Int)
    ts = range(0.0, stop=1.0, length=N)
    ω = 2π
    [SVector{d,Float64}(ntuple(i -> sin(ω * i * t), d)) for t in ts]
end

# --- Helpers ---

function flatten_signature(sig::Chen.Tensor{T}) where T
    d, m = sig.dim, sig.level
    out = T[]
    for k in 1:m
        start_idx = sig.offsets[k+1] + 1
        len = d^k
        append!(out, view(sig.coeffs, start_idx : start_idx+len-1))
    end
    return out
end

function compute_logsig_lyndon(path, m)
    d = length(path[1])
    tensor_type = Chen.Tensor{eltype(path[1])}
    
    # 1. Compute Signature
    sig = signature_path(tensor_type, path, m)
    
    # 2. Compute Log (Dense)
    log_sig = Chen.log(sig)
    
    # 3. Build Transform L (using internal Algebra function)
    # FIXED: Chen.build_L -> Chen.Algebra.build_L
    lynds, L, _ = Chen.Algebra.build_L(d, m)
    
    # 4. Project (using internal Algebra function)
    # FIXED: Chen.project_to_lyndon -> Chen.Algebra.project_to_lyndon
    return Chen.Algebra.project_to_lyndon(log_sig, lynds, L)
end

@testset "Correctness against iisignature" begin

    N = Fixtures.N

    @testset "d=2, m=4, Linear" begin
        d, m = 2, 4
        path = make_path_linear(d, N)
        
        sig_jl_tensor = signature_path(Chen.Tensor{Float64}, path, m)
        sig_jl_flat = flatten_signature(sig_jl_tensor)
        @test sig_jl_flat ≈ Fixtures.SIG_D2_M4_LINEAR atol=1e-12

        logsig_jl = compute_logsig_lyndon(path, m)
        @test logsig_jl ≈ Fixtures.LOGSIG_D2_M4_LINEAR atol=1e-12
    end

    @testset "d=2, m=4, Sin" begin
        d, m = 2, 4
        path = make_path_sin(d, N)

        sig_jl_tensor = signature_path(Chen.Tensor{Float64}, path, m)
        sig_jl_flat = flatten_signature(sig_jl_tensor)
        @test sig_jl_flat ≈ Fixtures.SIG_D2_M4_SIN atol=1e-12

        logsig_jl = compute_logsig_lyndon(path, m)
        @test logsig_jl ≈ Fixtures.LOGSIG_D2_M4_SIN atol=1e-12
    end

    @testset "d=3, m=3, Linear" begin
        d, m = 3, 3
        path = make_path_linear(d, N)

        sig_jl_tensor = signature_path(Chen.Tensor{Float64}, path, m)
        sig_jl_flat = flatten_signature(sig_jl_tensor)
        @test sig_jl_flat ≈ Fixtures.SIG_D3_M3_LINEAR atol=1e-12

        logsig_jl = compute_logsig_lyndon(path, m)
        @test logsig_jl ≈ Fixtures.LOGSIG_D3_M3_LINEAR atol=1e-12
    end

end