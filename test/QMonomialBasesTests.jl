module QMonomialBasesTests

using Test
using TensorPolynomialBases
using StaticArrays
using TensorValues

function _evaluate_1d!(v::AbstractMatrix{T},x,order,d) where T
  n = order + 1
  z = one(T)
  @inbounds v[d,1] = z
  @inbounds for i in 2:n
    v[d,i] = x[d]^(i-1)
  end
end

function _gradient_1d!(v::AbstractMatrix{T},x,order,d) where T
  n = order + 1
  z = zero(T)
  @inbounds v[d,1] = z
  @inbounds for i in 2:n
    v[d,i] = (i-1)*x[d]^(i-2)
  end
end

#function _evaluate_nd!(
#  v::AbstractVector{V},
#  x,
#  order,
#  terms::Vector{CartesianIndex{D}},
#  c::AbstractMatrix{T}) where {V,T,D}
#
#  dim = D
#  for d in 1:dim
#    _evaluate_1d!(c,x,order,d)
#  end
#
#  m = zero(_mutable(V))
#  o = one(T)
#  z = zero(T)
#  js = eachindex(m)
#  k = 1
#
#  for ci in terms
#
#    s = o
#    @inbounds for d in 1:dim
#      s *= c[d,ci[d]]
#    end
#
#    @inbounds for j in js
#      for i in js
#        m[i] = z
#      end
#      m[j] = s
#      v[k] = m
#      k += 1
#    end
#
#  end
#
#end

function _evaluate_nd!(
  v::AbstractVector{V},
  x,
  order,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T}) where {V,T,D}

  dim = D
  for d in 1:dim
    _evaluate_1d!(c,x,order,d)
  end

  o = one(T)
  k = 1

  for ci in terms

    s = o
    @inbounds for d in 1:dim
      s *= c[d,ci[d]]
    end

    k = _set_value!(v,s,k)

  end

end

@inline function _set_value!(v::AbstractVector{V},s::T,k) where {V,T}
  m = zero(_mutable(V))
  z = zero(T)
  js = eachindex(m)
  @inbounds for j in js
    @inbounds for i in js
      m[i] = z
    end
    m[j] = s
    v[k] = m
    k += 1
  end
  k
end

@inline function _set_value!(v::AbstractVector{<:Real},s,k)
    v[k] = s
    k+1
end

function _gradient_nd!(
  v::AbstractVector{G},
  x,
  order,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T},
  g::AbstractMatrix{T},
  ::Type{V}) where {G,T,D,V}

  dim = D
  for d in 1:dim
    _evaluate_1d!(c,x,order,d)
    _gradient_1d!(g,x,order,d)
  end

  z = zero(MVector{D,T})
  o = one(T)
  k = 1

  for ci in terms

    s = z
    for i in eachindex(s)
      s[i] = o
    end
    for q in 1:dim
      for d in 1:dim
        if d != q
          s[q] *= c[d,ci[d]]
        else
          s[q] *= g[d,ci[d]]
        end
      end
    end

    k = _set_gradient!(v,s,k,V)

  end

end

@inline function _set_gradient!(v::AbstractVector{G},s::MVector{D,T},k,::Type{<:Real}) where {G,D,T}
    v[k] = s
    k+1
end

@inline function _set_gradient!(v::AbstractVector{G},s::MVector{D,T},k,::Type{V}) where {V,G,D,T}
  m = zero(_mutable(G))
  w = zero(V)
  z = zero(T)
  for j in eachindex(w)
    @inbounds for i in eachindex(m)
      m[i] = z
    end
    for i in eachindex(s)
      m[i,j] = s[i]
    end
    v[k] = m
    k += 1
  end
  k
end

function _define_terms(filter,order,dim)
  n1d = order+1
  t = tuple(fill(n1d,dim)...)
  g = tuple(fill(1,dim)...)
  cis = CartesianIndices(t)
  co = CartesianIndex(g)
  [ ci for ci in cis if filter(Tuple(ci-co),order) ]
end

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
