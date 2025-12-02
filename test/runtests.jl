using Test
using ChenSignatures

@testset "ChenSignatures Tests" begin
    include("exp_log.jl")
    include("correctness.jl")
    include("test_float32.jl")
    include("test_add_scaled.jl")
    include("edge_cases.jl")
    include("test_enzyme.jl")
    include("chain_rules.jl")
end