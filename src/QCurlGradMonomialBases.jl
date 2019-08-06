
struct QCurlGradMonomialBasis{P,V,G,D} <: TensorPolynomialBasis{P,V,G}
  order::Int
  terms::CartesianIndices{D}
  perms::Matrix{Int}
end

function (::Type{QCurlGradMonomialBasis{P,V}})(order::Int) where {P,V}
  D = _length(P)
  @assert _length(V) == D
  G = _gradient_type(V,P)
  n1d = order
  t = fill(n1d,D)
  t[1] = order+1
  terms = CartesianIndices(tuple(t...))
  perms = _prepare_perms(D)
  QCurlGradMonomialBasis{P,V,G,D}(order,terms,perms)
end

# Implementation of the interface

length(b::QCurlGradMonomialBasis{P,V,G,D}) where {P,V,G,D} = D*(b.order+1)*(b.order)^(D-1)

ndims(b::QCurlGradMonomialBasis{P,V,G,D}) where {P,V,G,D} = D

function ScratchData(b::QCurlGradMonomialBasis{P,V}) where {P,V}
  T = eltype(V)
  dim = _length(P)
  n1d = b.order+1
  c = zeros(dim,n1d)
  g = zeros(dim,n1d)
  MonomialBasisCache{T}(c,g)
end

function evaluate!(
  v::AbstractVector{V},
  b::QCurlGradMonomialBasis{P,V,G,D},
  x::P,
  cache::MonomialBasisCache) where {P,V,G,D}
  _evaluate_nd_qgrad!(v,x,b.order,b.terms,b.perms,cache.c)
end

function gradient!(
  v::AbstractVector{G},
  b::QCurlGradMonomialBasis{P,V,G,D},
  x::P,
  cache::MonomialBasisCache) where {P,V,G,D}
  _gradient_nd_qgrad!(v,x,b.order,b.terms,b.perms,cache.c,cache.g,V)
end
