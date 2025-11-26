using ChenSignatures

# word_1 = Word([1,2,3])
word_2 = Word([1])
empty_word = Word()

tensor_1 = ChenSignatures.SparseTensor(Dict(empty_word=>1.0,word_2=>2.4), 3, 8)

tensor_2 = ChenSignatures.SparseTensor(Dict(empty_word=>1.0,word_2=>2.4), 3, 8)

result = ChenSignatures.SparseTensor(Dict{Word,Float64}(),3,8)

tensor_result = Tensor(result)
input_1 = Tensor(tensor_1)
input_2 = Tensor(tensor_2)

ChenSignatures.mul!(tensor_result, input_1, input_2)
ChenSignatures.mul!(result, tensor_1, tensor_2)

@show SparseTensor(tensor_result)
@show result

ChenSignatures.exp!(result,tensor_1)

vec = Vector{Float64}(undef, 3)
vec .= [2.0, 3.0, 4.5]
ChenSignatures.exp!(tensor_result, vec)
ChenSignatures.exp!(result,vec)
isapprox(SparseTensor(tensor_result),result)
