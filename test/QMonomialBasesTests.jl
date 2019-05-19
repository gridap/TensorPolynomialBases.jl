module QMonomialBasesTests

using Test
using TensorPolynomialBases
using StaticArrays
using TensorValues

using TensorPolynomialBases: _length, _gradient_type


order = 1
dim = 2
n1d = order + 1
v = zeros(dim,n1d)

x = SVector(2.0,3.0,4.0)

_evaluate_1d!(v,x,order,1)
_evaluate_1d!(v,x,order,2)
#_evaluate_1d!(v,x,order,3)

@show v

_gradient_1d!(v,x,order,1)
_gradient_1d!(v,x,order,2)
#_gradient_1d!(v,x,order,3)

@show v

import TensorPolynomialBases: _mutable

_mutable(::Type{T}) where T <:Real = T



filter(e,order) = true
#filter(e,order) = sum(e) <= order


terms = _define_terms(filter,order,dim)


V = SVector{2,Float64}
G = SMatrix{2,2,Float64,4}
n = length(V)*length(terms)
c = zeros(dim,n1d)
g = zeros(dim,n1d)
v = zeros(V,n)
h = zeros(G,n)
_evaluate_nd!(v,x,order,terms,c)
_gradient_nd!(h,x,order,terms,c,g,V)
@show v
@show h

V = Float64
G = SVector{2,Float64}
n = 1*length(terms)
c = zeros(dim,n1d)
g = zeros(dim,n1d)
v = zeros(V,n)
h = zeros(G,n)
_evaluate_nd!(v,x,order,terms,c)
_gradient_nd!(h,x,order,terms,c,g,V)
@show v
@show h



function run(n,v,x,order,terms,c)
  for i in 1:n
    _evaluate_nd!(v,x,order,terms,c)
  end
end


#@time run(10,v,x,order,terms,c)
#@time run(1000000,v,x,order,terms,c)





#order = 1
#
#P = SVector{2,Float64}
#V = SVector{1,Float64}
#
#basis = QMonomialBasis{P,V}(order)
#
#cache = ScratchData(basis)
#
#x = @SVector [1.0,2.0]
#
#@show evaluate(basis,x,cache)




end # module QMonomialBasesTests
