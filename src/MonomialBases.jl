
# Types and constructors

struct MonomialBasis{P,V,G,D} <: TensorPolynomialBasis{P,V,G}
  terms::Vector{CartesianIndex{D}}
  orders::NTuple{D,Int}
end

struct MonomialBasisCache{T}
  c::Matrix{T}
  g::Matrix{T}
end

function (::Type{MonomialBasis{P,V}})(filter::Function,order::Int) where {P,V}
  D = _length(P)
  terms = _define_terms(filter,order,D)
  G = _gradient_type(V,P)
  orders = tuple(fill(order,D)...)
  MonomialBasis{P,V,G,D}(terms,orders)
end

function (::Type{MonomialBasis{P,V}})(orders::NTuple{N,Int}) where {P,V,N}
  terms = [ ci for ci in CartesianIndices(orders.+1) if true]
  D = _length(P)
  @assert D == N
  G = _gradient_type(V,P)
  MonomialBasis{P,V,G,D}(terms,orders)
end

# Implementation of the interface

length(b::MonomialBasis{P,V}) where {P,V} = _length(V)*length(b.terms)

ndims(b::MonomialBasis{P}) where P = _length(P)

function ScratchData(b::MonomialBasis{P,V}) where {P,V}
  T = eltype(V)
  dim = _length(P)
  n1d = maximum(b.orders.+1)
  c = zeros(dim,n1d)
  g = zeros(dim,n1d)
  MonomialBasisCache{T}(c,g)
end

function evaluate!(
  v::AbstractVector{V},
  b::MonomialBasis{P,V},
  x::P,
  cache::MonomialBasisCache) where {P,V}
  _evaluate_nd!(v,x,b.orders,b.terms,cache.c)
end

function gradient!(
  v::AbstractVector{G},
  b::MonomialBasis{P,V,G},
  x::P,
  cache::MonomialBasisCache) where {P,V,G}
  _gradient_nd!(v,x,b.orders,b.terms,cache.c,cache.g,V)
end

# Helpers

_q_filter(e,o) = true

function _define_terms(filter,order,dim)
  n1d = order+1
  t = tuple(fill(n1d,dim)...)
  g = tuple(fill(1,dim)...)
  cis = CartesianIndices(t)
  co = CartesianIndex(g)
  [ ci for ci in cis if filter(Tuple(ci-co),order) ]
end

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

function _evaluate_nd!(
  v::AbstractVector{V},
  x,
  orders,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T}) where {V,T,D}

  dim = D
  for d in 1:dim
    _evaluate_1d!(c,x,orders[d],d)
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
  orders,
  terms::AbstractVector{CartesianIndex{D}},
  c::AbstractMatrix{T},
  g::AbstractMatrix{T},
  ::Type{V}) where {G,T,D,V}

  dim = D
  for d in 1:dim
    _evaluate_1d!(c,x,orders[d],d)
    _gradient_1d!(g,x,orders[d],d)
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

@inline function _set_gradient!(
  v::AbstractVector{G},s::MVector{D,T},k,::Type{<:Real}) where {G,D,T}

  v[k] = s
  k+1
end

@inline function _set_gradient!(
  v::AbstractVector{G},s::MVector{D,T},k,::Type{V}) where {V,G,D,T}

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

