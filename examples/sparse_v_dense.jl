using Chen

# word_1 = Word([1,2,3])
word_2 = Word([1])
empty_word = Word()

tensor_1 = Chen.SparseTensor(Dict(empty_word=>1.0,word_2=>2.4), 3, 8)

tensor_2 = Chen.SparseTensor(Dict(empty_word=>1.0,word_2=>2.4), 3, 8)

result = Chen.SparseTensor(Dict{Word,Float64}(),3,8)

tensor_result = Tensor(result)
input_1 = Tensor(tensor_1)
input_2 = Tensor(tensor_2)

Chen.mul!(tensor_result, input_1, input_2)
Chen.mul!(result, tensor_1, tensor_2)

@show SparseTensor(tensor_result)
@show result

Chen.exp!(result,tensor_1)

vec = Vector{Float64}(undef, 3)
vec .= [2.0, 3.0, 4.5]
Chen.exp!(tensor_result, vec)
Chen.exp!(result,vec)
isapprox(SparseTensor(tensor_result),result)
