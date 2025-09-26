using Test
using PathSignatures

@testset "PathSignatures – basic ops" begin
    # Setup
    word_1 = Word([1, 2, 3])  # kept to ensure Word accepts longer indices
    word_2 = Word([1])
    empty_word = Word()

    t1 = PathSignatures.SparseTensor(Dict(empty_word => 1.0, word_2 => 2.4), 3, 8)
    t2 = PathSignatures.SparseTensor(Dict(empty_word => 1.0, word_2 => 2.4), 3, 8)

    # Prepare result buffers
    res = PathSignatures.SparseTensor(Dict{Word,Float64}(), 3, 8)

    # Tensor views
    i1 = Tensor(t1)
    i2 = Tensor(t2)
    tres = Tensor(res)

    # ---- mul! consistency (Tensor vs SparseTensor destinations) ----
    PathSignatures.mul!(tres, i1, i2)
    PathSignatures.mul!(res,  t1, t2)

    @test SparseTensor(tres) == res

    # ---- exp! on a SparseTensor from another SparseTensor ----
    # (Just ensure it runs and produces a SparseTensor with the same shape domain)
    exp_from_sparse = PathSignatures.SparseTensor(Dict{Word,Float64}(), 3, 8)
    PathSignatures.exp!(exp_from_sparse, t1)
    @test isa(exp_from_sparse, typeof(t1))
    @test exp_from_sparse.n == t1.n && exp_from_sparse.m == t1.m

    # ---- exp! with a vector into Tensor vs SparseTensor; results should match ----
    vec = [2.0, 3.0, 4.5]

    # fresh buffers so this section is independent
    res_vec  = PathSignatures.SparseTensor(Dict{Word,Float64}(), 3, 8)
    tres_vec = Tensor(PathSignatures.SparseTensor(Dict{Word,Float64}(), 3, 8))

    PathSignatures.exp!(tres_vec, vec)
    PathSignatures.exp!(res_vec,  vec)

    @test isapprox(SparseTensor(tres_vec), res_vec; rtol=1e-12, atol=1e-12)

    # Sanity: word_1 is valid and doesn’t break construction
    @test isa(word_1, Word)
end
