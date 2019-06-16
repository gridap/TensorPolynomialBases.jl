module MonomialBasesTests

using Test
using TensorPolynomialBases
using StaticArrays
using TensorValues

# For StaticArrays

P = SVector{2,Float64}
V = SVector{3,Float64}
filter(e,order) = true
order = 1
basis = MonomialBasis{P,V}(filter,order)

G = gradient_type(basis)

v = V[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
      [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0],
      [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0],
      [6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]]

g = G[[0.0 0.0 0.0; 0.0 0.0 0.0], [0.0 0.0 0.0; 0.0 0.0 0.0],
      [0.0 0.0 0.0; 0.0 0.0 0.0], [1.0 0.0 0.0; 0.0 0.0 0.0],
      [0.0 1.0 0.0; 0.0 0.0 0.0], [0.0 0.0 1.0; 0.0 0.0 0.0],
      [0.0 0.0 0.0; 1.0 0.0 0.0], [0.0 0.0 0.0; 0.0 1.0 0.0],
      [0.0 0.0 0.0; 0.0 0.0 1.0], [3.0 0.0 0.0; 2.0 0.0 0.0],
      [0.0 3.0 0.0; 0.0 2.0 0.0], [0.0 0.0 3.0; 0.0 0.0 2.0]]

x = SVector(2.0,3.0)
test_polynomial_basis(basis,x,v,g)

# For TensorValues

P = VectorValue{2,Float64}
V = VectorValue{3,Float64}
filter(e,order) = true
order = 1
basis = MonomialBasis{P,V}(filter,order)

G = gradient_type(basis)

v = reinterpret(V,v)
g = reinterpret(G,g)

x = VectorValue(2.0,3.0)
test_polynomial_basis(basis,x,v,g)

# Idem but with a wither set of inputs

P = VectorValue{2}
basis = MonomialBasis{P,V}(filter,order)

x = VectorValue(2,3)
test_polynomial_basis(basis,x,v,g)

# Q space with isotropic order

P = VectorValue{2}
V = VectorValue{3,Float64}
orders = (1,1,1)
basis = MonomialBasis{P,V}(orders)

x = VectorValue(2.0,3.0)
test_polynomial_basis(basis,x,v,g)

# For Reals (SVector point)

P = SVector{2,Float64}
V = Float64
filter(e,order) = sum(e) <= order
order = 1
basis = MonomialBasis{P,V}(filter,order)

G = gradient_type(basis)
@test G <: SArray

v = V[1.0, 2.0, 3.0]
g = G[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

x = SVector(2.0,3.0)
test_polynomial_basis(basis,x,v,g)

# For Reals (VectorValue point)

P = VectorValue{2,Float64}
V = Float64
filter(e,order) = sum(e) <= order
order = 1
basis = MonomialBasis{P,V}(filter,order)

G = gradient_type(basis)
@test G <: VectorValue

v = V[1.0, 2.0, 3.0]
g = G[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

x = VectorValue(2.0,3.0)
test_polynomial_basis(basis,x,v,g)

# For StaticArrays

P = SVector{2,Float64}
V = SMatrix{3,1,Float64,3}
filter(e,order) = sum(e) <= order
order = 1
basis = MonomialBasis{P,V}(filter,order)

x = SVector(2.0,3.0)

v = V[[1.0; 0.0; 0.0], [0.0; 1.0; 0.0], [0.0; 0.0; 1.0],
      [2.0; 0.0; 0.0], [0.0; 2.0; 0.0], [0.0; 0.0; 2.0],
      [3.0; 0.0; 0.0], [0.0; 3.0; 0.0], [0.0; 0.0; 3.0]]

G = gradient_type(basis)
g = G[[0.0 0.0 0.0; 0.0 0.0 0.0], [0.0 0.0 0.0; 0.0 0.0 0.0],
      [0.0 0.0 0.0; 0.0 0.0 0.0], [1.0 0.0 0.0; 0.0 0.0 0.0],
      [0.0 1.0 0.0; 0.0 0.0 0.0], [0.0 0.0 1.0; 0.0 0.0 0.0],
      [0.0 0.0 0.0; 1.0 0.0 0.0], [0.0 0.0 0.0; 0.0 1.0 0.0],
      [0.0 0.0 0.0; 0.0 0.0 1.0]]

x = SVector(2.0,3.0)
test_polynomial_basis(basis,x,v,g)


end # module MonomialBasesTests
