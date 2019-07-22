
# Types and constructors

struct QGradMonomialBasis{P,G,D} <: TensorPolynomialBasis{P,P,G}
  order::Int
  terms::CartesianIndices{D}
  perms::Matrix{Int}
end

function (::Type{QGradMonomialBasis{P}})(order::Int) where P
  D = _length(P)
  G = _gradient_type(P,P)
  n1d = order+1
  t = fill(n1d,D)
  t[1] = order
  terms = CartesianIndices(tuple(t...))
  perms = _prepare_perms(D)
  QGradMonomialBasis{P,G,D}(order,terms,perms)
end

# Implementation of the interface

length(b::QGradMonomialBasis{P,G,D}) where {P,G,D} = D*b.order*(b.order+1)^(D-1)

ndims(b::QGradMonomialBasis{P,G,D}) where {P,G,D} = D

function ScratchData(b::QGradMonomialBasis{P}) where P
  V = P
  T = eltype(V)
  dim = _length(P)
  n1d = b.order+1
  c = zeros(dim,n1d)
  g = zeros(dim,n1d)
  MonomialBasisCache{T}(c,g)
end

function evaluate!(
  v::AbstractVector{P},
  b::QGradMonomialBasis{P,G,D},
  x::P,
  cache::MonomialBasisCache) where {P,G,D}
  _evaluate_nd_qgrad!(v,x,b.order,b.terms,b.perms,cache.c)
end

function gradient!(
  v::AbstractVector{G},
  b::QGradMonomialBasis{P,G,D},
  x::P,
  cache::MonomialBasisCache) where {P,G,D}
  V = P
  _gradient_nd_qgrad!(v,x,b.order,b.terms,b.perms,cache.c,cache.g,V)
end

# Helpers

function _prepare_perms(D)
  perms = zeros(Int,D,D)
  for j in 1:D
    for d in j:D
      perms[d,j] =  d-j+1
    end
    for d in 1:(j-1)
      perms[d,j] =  d+(D-j)+1
    end
  end
  perms
end

function _evaluate_nd_qgrad!(
  v::AbstractVector{V},
  x,
  order,
  terms::CartesianIndices{D},
  perms::Matrix{Int},
  c::AbstractMatrix{T}) where {V,T,D}

  dim = D
  for d in 1:dim
    _evaluate_1d!(c,x,order,d)
  end

  o = one(T)
  k = 1
  m = zero(_mutable(V))
  js = eachindex(m)
  z = zero(T)

  for ci in terms

    for j in js

      @inbounds for i in js
        m[i] = z
      end

      s = o
      @inbounds for d in 1:dim
        s *= c[d,ci[perms[d,j]]]
      end

      m[j] = s
      v[k] = m
      k += 1

    end

  end

end

function _gradient_nd_qgrad!(
  v::AbstractVector{G},
  x,
  order,
  terms::CartesianIndices{D},
  perms::Matrix{Int},
  c::AbstractMatrix{T},
  g::AbstractMatrix{T},
  ::Type{V}) where {G,T,D,V}

  dim = D
  for d in 1:dim
    _evaluate_1d!(c,x,order,d)
    _gradient_1d!(g,x,order,d)
  end

  z = zero(_mutable(V))
  m = zero(_mutable(G))
  js = eachindex(z)
  mjs = eachindex(m)
  o = one(T)
  zi = zero(T)
  k = 1

  for ci in terms

    for j in js

      s = z
      for i in js
        s[i] = o
      end

      for q in 1:dim
        for d in 1:dim
          if d != q
            s[q] *= c[d,ci[perms[d,j]]]
          else
            s[q] *= g[d,ci[perms[d,j]]]
          end
        end
      end

      @inbounds for i in mjs
        m[i] = zi
      end

      for i in js
        m[i,j] = s[i]
      end
      v[k] = m
      k += 1

    end

  end

end
