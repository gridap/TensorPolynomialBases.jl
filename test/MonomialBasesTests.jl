module MonomialBasesTests

using Test
using TensorPolynomialBases
using StaticArrays
using TensorValues

P = SVector{2,Float64}
V = SVector{3,Float64}
filter(e,order) = true
order = 1
basis = MonomialBasis{P,V}(filter,order)

cache = ScratchData(basis)

G = gradient_type(basis)
v = zeros(V,length(basis))
g = zeros(G,length(basis))

x = SVector(2.0,3.0)

evaluate!(v,basis,x,cache)
gradient!(g,basis,x,cache)

@show v
@show g

end # module MonomialBasesTests
