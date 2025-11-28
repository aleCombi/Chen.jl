# test/test_enzyme_module.jl

using ChenSignatures
using Enzyme
using FiniteDifferences

println("=" ^ 70)
println("ENZYME: Does module function work now?")
println("=" ^ 70)

function test_it()
    function loss(x::Vector{Float64})
        D, M = 3, 3
        out = ChenSignatures.Tensor{Float64, D, M}()
        ChenSignatures.non_generated_exp!(out, x)
        return sum(out.coeffs)
    end
    
    x = [0.1, 0.2, 0.15]
    
    println("Forward: $(loss(x))")
    println("Starting Enzyme.autodiff...")
    
    grad = zeros(3)
    try
        Enzyme.autodiff(Reverse, loss, Active, Duplicated(x, grad))
        println("✅ SUCCESS: $grad")
        
        fdm = central_fdm(5, 1)
        grad_fd = FiniteDifferences.grad(fdm, loss, x)[1]
        println("FiniteDiff: $grad_fd")
        println("Error: $(maximum(abs.(grad .- grad_fd)))")
        
    catch e
        println("❌ CRASHED: $e")
    end
end

test_it()