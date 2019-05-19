module InterfacesTests

using Test
using TensorPolynomialBases
using StaticArrays

P = SVector{2,Float64}
V = SVector{3,Float64}
G = SMatrix{2,3,Float64,6}
filter(e,order) = sum(e) <= order
order = 1
basis = MonomialBasis{P,V}(filter,order)

x = SVector(2.0,3.0)

cache = ScratchData(basis)
v = zeros(V,length(basis))
g = zeros(G,length(basis))
evaluate!(v,basis,x,cache)
gradient!(g,basis,x,cache)
test_polynomial_basis(basis,x,v,g)

v = evaluate(basis,x)
g = gradient(basis,x)
test_polynomial_basis(basis,x,v,g)

@test P == point_type(basis)
@test V == value_type(basis)
@test G == gradient_type(basis)

end # module InterfacesTests
