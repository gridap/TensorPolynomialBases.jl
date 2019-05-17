
mutable struct QScratchData{T,P,V,G}
  order::Int
  dim::Int
  l::Int
  x::Vector{P}
  v::Matrix{V}
  g::Matrix{G}
  cv::Array{T,3}
  cg::Array{T,3}
end

function (::Type{QScratchData{T,P,V,G}})(order::Int,dim::Int,l::Int) where {T,P,V,G}
  x = zeros(P,1)
  v = zeros(P,(l,1))
  g = zeros(G,(l,1))
  cv = zeros(T,(1,l,0))
  cg = zeros(T,(dim,l,0))
  QScratchData{T,P,V,G}(order,dim,l,x,v,g,cv,cg)
end

function npoints!(cache::QScratchData{T},n::Int) where T
  dim, l, m = size(cache.cg)
  if n > m
    cache.cv = zeros(T,(1,l,n))
    cache.cg = zeros(T,(dim,l,n))
  end
end

"""
Optimized implementation of a monomial basis of the Q-space
"""
struct QMonomialBasis{P,V,G} <: TensorPolynomialBasis{P,V,G}
  order::Int
end

function (::Type{QMonomialBasis{P,V}})(order::Int) where {P,V}
  G = _gradient_type(V,P)
  QMonomialBasis{P,V,G}(order)
end

function length(b::QMonomialBasis{P,V}) where {P,V}
  dim = _length(P)
  n = _length(V)
  n *(( b.order + 1)^dim)
end

ndims(b::QMonomialBasis{P}) where P = _length(P)

function ScratchData(b::QMonomialBasis{P,V}) where {P,V}
  order = b.order
  dim = ndims(b)
  l = length(b)
  T = eltype(V)
  QScratchData{T,P,V,G}(order,dim,l)
end

function evaluate!(
  v::AbstractVector{V},
  b::QMonomialBasis{P,V},
  x::P,
  cache::QScratchData) where {P,V}
  cache.x[1] = x
  evaluate!(cache.v,b,cache.x,cache)
  v[:] .= cache.v[:,1]
end

function gradient!(
  v::AbstractVector{V},
  b::QMonomialBasis{P,V},
  x::P,
  cache::QScratchData) where {P,V}
  cache.cx[1] = x
  evaluate!(cache.g,b,cache.x,cache)
  v[:] .= cache.g[:,1]
end

function evaluate!(
  v::AbstractMatrix{V},
  b::QMonomialBasis{P,V},
  x::AbstractVector{P},
  cache::QScratchData) where {P,V}

  npoints!(cache,length(x))

  _evaluate_nd_tensor!(v,x,b.order,cache.cv)

end

function gradient!(
  v::AbstractMatrix{G},
  b::QMonomialBasis{P,V,G},
  x::AbstractVector{P},
  cache::QScratchData) where {P,V,G}

  npoints!(cache,length(x))

  _gradient_nd_tensor!(v,x,b.order,cache.cg)

end

# Helpers

function _evaluate_1d!(
  v::AbstractArray{T,3},x::AbstractVector,order::Int,d,k) where T
  n = order + 1
  o = one(T)
  for (j,p) in enumerate(x)
    v[k,1,j] *= o
    for i in 2:n
      v[k,i,j] *= p[d]^(i-1)
    end
  end
end

function _gradient_1d!(
  v::AbstractArray{T,3},x::AbstractVector,order::Int,d,k) where T
  n = order + 1
  z = zero(T)
  for (j,p) in enumerate(x)
    v[k,1,j] *= z
    for i in 2:n
      v[k,i,j] *= (i-1)*p[d]^(i-2)
    end
  end
end

function _evaluate_nd_scalar!(
  v::AbstractArray{T,3}, x::AbstractVector{P}, order::Int) where {P,T}
  v .= one(T)
  dim = _length(P)
  for d in 1:dim
    _evaluate_1d!(v,x,order,d,1)
  end
end

function _gradient_nd_scalar!(
  v::AbstractArray{T,3}, x::AbstractVector{P}, order::Int) where {P,T}
  dim = _length(P)
  v .= one(T)
  for dj in 1:dim
    for di in 1:dim
      if di != dj
        _evaluate_1d!(v,x,order,di,dj)
      else
        _gradient_1d!(v,x,order,di,dj)
      end
    end
  end
end

function _evaluate_nd_tensor!(
  v::AbstractMatrix{V},
  x::AbstractVector{P},
  order::Int,
  c::AbstractArray{T,3}) where {P,V,T}
  
  _evaluate_nd_scalar!(c,x,order)

  _fill_value!(v,c)

end

function _gradient_nd_tensor!(
  v::AbstractMatrix{V},
  x::AbstractVector{P},
  order::Int,
  c::AbstractArray{T,3}) where {P,V,T}
  
  _gradient_nd_scalar!(c,x,order)

  _fill_gradient!(v,c)

end

_length(x) = length(x)

_length(::Type{<:Real}) = 1

function _fill_value!(
  v::AbstractMatrix{V},
  s::AbstractArray{T,3}) where {V,T}

  m = zero(_mutable(V))
  for p in 1:size(s,3)
    k = 1
    for i in 1:size(s,2)
      si = s[1,i,p]
      for j in eachindex(m)
        m .= 0.0
        m[j] = si
        v[k,p] = m
        k += 1
      end
    end
  end
end

function _fill_gradient!(
  v::AbstractMatrix{G},
  s::AbstractArray{T,3},
  ::Type{V}) where {T,V,G}

  m = zero(_mutable(G))
  w = zero(V)
  for p in 1:size(s,3)
    k = 1
    for j in 1:size(s,2)
      for l in CartesianIndices(w)
        m .= 0.0
        for i in 1:size(s,1)
          m[i,l] = s[i,j,p]
        end
        v[k,p] = m
        k += 1
      end 
    end
  end
end

function _fill_gradient_scalar!(
  v::AbstractMatrix{G},
  s::AbstractArray{T,3}) where {T,V,G}

  m = zero(_mutable(G))
  for p in 1:size(s,3)
    k = 1
    for j in 1:size(s,2)
      m .= 0.0
      for i in 1:size(s,1)
        m[i,l] = s[i,j,p]
      end
      v[k,p] = m
      k += 1
    end
  end
end

