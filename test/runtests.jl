using Test
using Chen

@testset "Chen Tests" begin
    include("exp_log.jl")
    include("correctness.jl")
    include("test_float32.jl")
    include("edge_cases.jl")
end