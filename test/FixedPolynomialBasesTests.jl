module FixedPolynomialBasesTests

using TensorPolynomialBases
using Test
import FixedPolynomials; const fp = FixedPolynomials
using StaticArrays
using TensorValues

filter(e,O) = (sum(e) <= O)

order = 3
dim = 2
B = fp.System{Float64}(filter,order,dim)

cfg = fp.JacobianConfig(B)

x = [1.0,2.0]

@test fp.evaluate(B,x,cfg) ≈ [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 8.0]
@test fp.jacobian(B,x,cfg) ≈ [0.0 0.0; 1.0 0.0; 2.0 0.0; 3.0 0.0; 0.0 1.0;
                     2.0 1.0; 4.0 1.0; 0.0 4.0; 4.0 4.0; 0.0 12.0]

order = 1
dims = 2

filter(e,O) = true

T = Float64
V = SVector{3,Float64}
basis = FixedPolynomialBasis{T,V}(filter,order,dim)

n = length(basis)
v = zeros(V,n)
G = gradient_type(basis)
w = zeros(G,n)
x = SVector(2.0,3.0)

cache = ScratchData(basis)

evaluate!(v,basis,x,cache)

gradient!(w,basis,x,cache)

r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [2.0, 0.0, 0.0],
     [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0],
     [0.0, 0.0, 3.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]]

@test v == r

v = evaluate(basis,x,cache)

@test v == r

@test evaluate(basis,[x,x],cache) == hcat(v,v)

r = [[0.0 0.0 0.0; 0.0 0.0 0.0], [0.0 0.0 0.0; 0.0 0.0 0.0],
     [0.0 0.0 0.0; 0.0 0.0 0.0], [1.0 0.0 0.0; 0.0 0.0 0.0],
     [0.0 1.0 0.0; 0.0 0.0 0.0], [0.0 0.0 1.0; 0.0 0.0 0.0],
     [0.0 0.0 0.0; 1.0 0.0 0.0], [0.0 0.0 0.0; 0.0 1.0 0.0],
     [0.0 0.0 0.0; 0.0 0.0 1.0], [3.0 0.0 0.0; 2.0 0.0 0.0],
     [0.0 3.0 0.0; 0.0 2.0 0.0], [0.0 0.0 3.0; 0.0 0.0 2.0]]

@test w == r

w = gradient(basis,x,cache)

@test w == r

@test gradient(basis,[x,x],cache) == hcat(w,w)

T = Float64
V = VectorValue{3,T}
basis = FixedPolynomialBasis{T,V}(filter,order,dim)
G = gradient_type(basis)

cache = ScratchData(basis)

@test evaluate(basis,x,cache) == reinterpret(V,v)

@test gradient(basis,x,cache) == reinterpret(G,w)

T = Float64
basis = FixedPolynomialBasis{T,T,SVector{dim,T}}(filter,order,dim)

n = length(basis)
V = value_type(basis)
v = zeros(V,n)
G = gradient_type(basis)
w = zeros(G,n)
x = SVector(2.0,3.0)

cache = ScratchData(basis)

evaluate!(v,basis,x,cache)

gradient!(w,basis,x,cache)

r = [1.0, 2.0, 3.0, 6.0]

@test v == r

r = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [3.0, 2.0]]

@test w == r



end #module FixedPolynomialBasesTests
