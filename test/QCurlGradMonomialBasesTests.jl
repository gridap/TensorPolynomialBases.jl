module QCurlGradMonomialBasesTests

using Test
using TensorPolynomialBases
using TensorValues

P = VectorValue{3}
V = VectorValue{3,Float64}
order = 1
basis = QCurlGradMonomialBasis{P,V}(order)
G = gradient_type(basis)

x = VectorValue(2,3,5)

v = V[
  (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
  (2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 5.0)]

g = G[
  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
  (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
  (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)]

test_polynomial_basis(basis,x,v,g)

P = VectorValue{2}
V = VectorValue{2,Float64}
order = 2
basis = QCurlGradMonomialBasis{P,V}(order)
G = gradient_type(basis)

x = VectorValue(2,3)

v = V[
  (1.0, 0.0), (0.0, 1.0), (2.0, 0.0), (0.0, 3.0),
  (4.0, 0.0), (0.0, 9.0), (3.0, 0.0), (0.0, 2.0),
  (6.0, 0.0), (0.0, 6.0), (12.0, 0.0), (0.0, 18.0)]

g = G[
  (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0),
  (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
  (4.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 6.0),
  (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0),
  (3.0, 2.0, 0.0, 0.0), (0.0, 0.0, 3.0, 2.0),
  (12.0, 4.0, 0.0, 0.0),(0.0, 0.0, 9.0, 12.0)]

test_polynomial_basis(basis,x,v,g)

end # module
