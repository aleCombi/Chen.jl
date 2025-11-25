using Test
using Chen

@testset "Chen Tests" begin
    include("exp_log.jl")
    include("correctness.jl")
end