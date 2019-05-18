
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

function (::Type{QScratchData{T,P,V,G}})(
  order::Int,dim::Int,l::Int) where {T,P,V,G}
  n1d = (order+1)
  x = zeros(P,1)
  v = zeros(V,(l,1))
  g = zeros(G,(l,1))
  cv = zeros(T,(dim,n1d,0))
  cg = zeros(T,(dim,n1d,0))
  QScratchData{T,P,V,G}(order,dim,l,x,v,g,cv,cg)
end

function npoints!(cache::QScratchData{T},npoints::Int) where T
  dim, n1d, m = size(cache.cg)
  if npoints > m
    cache.cv = zeros(T,(dim,n1d,npoints))
    cache.cg = zeros(T,(dim,n1d,npoints))
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

function ScratchData(b::QMonomialBasis{P,V,G}) where {P,V,G}
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
  v::AbstractArray{T,3},x::AbstractVector,order::Int,d) where T
  n = order + 1
  o = one(T)
  for (j,p) in enumerate(x)
    v[d,1,j] = o
    for i in 2:n
      v[d,i,j] = p[d]^(i-1)
    end
  end
end

function _gradient_1d!(
  v::AbstractArray{T,3},x::AbstractVector,order::Int,d) where T
  n = order + 1
  z = zero(T)
  for (j,p) in enumerate(x)
    v[d,1,j] = z
    for i in 2:n
      v[d,i,j] = (i-1)*p[d]^(i-2)
    end
  end
end

function _evaluate_nd_scalar!(
  v::AbstractArray{T,3}, x::AbstractVector{P}, order::Int) where {P,T}
  dim = _length(P)
  for d in 1:dim
    _evaluate_1d!(v,x,order,d)
  end
end

function _gradient_nd_scalar!(
  v::AbstractArray{T,3}, x::AbstractVector{P}, order::Int) where {P,T}
  dim = _length(P)
  for d in 1:dim
    _gradient_1d!(v,x,order,d)
  end
end


function _evaluate_nd_tensor!(
  v::AbstractMatrix{V},
  x::AbstractVector{P},
  order::Int,
  c::AbstractArray{T,3}) where {P,V,T}
  
  _evaluate_nd_scalar!(c,x,order)

  _fill_value!(v,c,length(x))

end

function _gradient_nd_tensor!(
  v::AbstractMatrix{V},
  x::AbstractVector{P},
  order::Int,
  c::AbstractArray{T,3}) where {P,V,T}
  
  _gradient_nd_scalar!(c,x,order)

  _fill_gradient!(v,c,length(x))

end

_length(x) = length(x)

_length(::Type{<:Real}) = 1

function _fill_value!(
  v::AbstractMatrix{V},
  s::AbstractArray{T,3},
  npoints::Int) where {V,T}

  m = zero(_mutable(V))
  dim, n1d, _ = size(s)
  cis = CartesianIndices((dim,n1d))
  for p in 1:npoints
    k = 1
    for ci in cis
      si = s[ci,p]
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
  ::Type{V},
  npoints::Int) where {T,V,G}

  dim, n1d, _ = size(s)
  cis = CartesianIndices((dim,n1d))

  m = zero(_mutable(G))
  w = zero(V)
  ls = CartesianIndices(w)
  for p in 1:npoints
    k = 1
    for ci in cis
      si = s[ci,p]
      for l in ls
        m .= 0.0
        for i in 1:size(s,1)
          m[i,l] = s[i,j,p]
        end
        v[k,p] = m
        k += 1
      end 




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
  s::AbstractArray{T,3},
  npoints::Int) where {T,V,G}

  m = zero(_mutable(G))
  for p in 1:npoints
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

