module TensorPolynomialBasesTests

using TensorPolynomialBases
using Test
import FixedPolynomials; const fp = FixedPolynomials
using StaticArrays

filter(e,O) = (sum(e) <= O)

order = 3
dim = 2
B = fp.System{Float64}(filter,order,dim)

cfg = fp.JacobianConfig(B)

x = [1.0,2.0]

@test fp.evaluate(B,x,cfg) ≈ [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 8.0]
@test fp.jacobian(B,x,cfg) ≈ [0.0 0.0; 1.0 0.0; 2.0 0.0; 3.0 0.0; 0.0 1.0;
                     2.0 1.0; 4.0 1.0; 0.0 4.0; 4.0 4.0; 0.0 12.0]


@test gradient_type(SVector{3,Int},Val(2)) == SMatrix{2,3,Int,6}

@test gradient_type(SMatrix{3,4,Int},Val(2)) == SArray{Tuple{2,3,4},Int,3,24}

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

r = [[0.0 0.0 0.0; 0.0 0.0 0.0], [0.0 0.0 0.0; 0.0 0.0 0.0],
     [0.0 0.0 0.0; 0.0 0.0 0.0], [1.0 0.0 0.0; 0.0 0.0 0.0],
     [0.0 1.0 0.0; 0.0 0.0 0.0], [0.0 0.0 1.0; 0.0 0.0 0.0],
     [0.0 0.0 0.0; 1.0 0.0 0.0], [0.0 0.0 0.0; 0.0 1.0 0.0],
     [0.0 0.0 0.0; 0.0 0.0 1.0], [3.0 0.0 0.0; 2.0 0.0 0.0],
     [0.0 3.0 0.0; 0.0 2.0 0.0], [0.0 0.0 3.0; 0.0 0.0 2.0]]

@test w == r

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

end # module TensorPolynomialBasesTests
