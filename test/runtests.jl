using Test
using ChenSignatures

@testset "ChenSignatures Tests" begin
    include("exp_log.jl")
    include("correctness.jl")
    include("test_float32.jl")
    include("edge_cases.jl")
    include("test_enzyme.jl")
end