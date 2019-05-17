
"""
Create a FixedPolynomials.System object representing a monomial basis. 
"""
function (::Type{fp.System{T}})(filter::Function, order::Int, dim::Int) where T
  B = _setup_system(filter,order,dim,T)
end

function _setup_system(filter,O,D,T)
  @polyvar x[1:D]
  o = CartesianIndex(ones(Int,D)...)
  cs = CartesianIndices( tuple(fill(O+1,D)...) )
  A = fp.Polynomial{T}[]
  for c in cs
    e = Tuple(c-o)
    if filter(e,O)
      f = 1
      for (xi,p) in zip(x,e)
        f *= xi^p
      end
      poly = fp.Polynomial{T}(f)
      push!(A,poly)
    end
  end
  fp.System(A)
end

function fp.evaluate!(
  v::AbstractVector{T},
  s::fp.System, 
  x::VectorValue{D,T},
  cfg::fp.JacobianConfig) where {D,T}
  fp.evaluate!(v,s,x.array,cfg)
end

function fp.jacobian!(
  v::AbstractMatrix{T},
  s::fp.System, 
  x::VectorValue{D,T},
  cfg::fp.JacobianConfig) where {D,T}
  fp.jacobian!(v,s,x.array,cfg)
end

"""
Scratch data associated with FixedPolynomialBasis
"""
struct FixedScratchData{T,V,G,C<:fp.JacobianConfig}
  cfg::C
  value::Vector{T}
  jacob::Matrix{T}
  v::Vector{V}
  g::Vector{G}
end

function (::Type{FixedScratchData{T,V,G}})(system::fp.System{T}) where {T,V,G}
  cfg = fp.JacobianConfig(system)
  n = length(system)
  d = fp.nvariables(system)
  value = zeros(T,n)
  jacob = zeros(T,(n,d))
  m = length(zero(V))*n
  v = zeros(V,m)
  g = zeros(G,m)
  FixedScratchData(cfg,value,jacob,v,g)
end

"""
Implementation of a tensor-valued multivariate polynomial basis
using the functionality given by the package FixedPolynomials
"""
struct FixedPolynomialBasis{P,V,G,T} <: TensorPolynomialBasis{P,V,G}
  system::fp.System{T}
end

"""
Generates a `FixedPolynomialBasis` object, to be evaluated at objects 
of type `T` and has value of type `V`
"""
function (::Type{FixedPolynomialBasis{P,V}})(
  filter::Function,order::Int) where {P,V}
  G = _gradient_type(V,P)
  FixedPolynomialBasis{P,V,G}(filter,order)
end

"""
Generates a `FixedPolynomialBasis` object, whose coefficients
are of type `T`, the value of type `V`, and the gradient of type G
"""
function (::Type{FixedPolynomialBasis{P,V,G}})(
  filter::Function, order::Int) where {P<:PointType{D},V,G} where D
  T = eltype(P)
  system = fp.System{T}(filter,order,D)
  FixedPolynomialBasis{P,V,G,T}(system)
end

function length(b::FixedPolynomialBasis{P,V}) where {P,V}
  length(zero(V)) * length(b.system)
end

ndims(b::FixedPolynomialBasis) = nvariables(b.system)

function ScratchData(b::FixedPolynomialBasis{P,V,G,T}) where {P,V,G,T}
  FixedScratchData{T,V,G}(b.system)
end

# Single-point versions of evaluate! and gradient!

function evaluate!(
  v::AbstractVector{V},
  b::FixedPolynomialBasis{P,V,G,T},
  x::P,
  cache::FixedScratchData{T}) where {P,V,G,T}

  fp.evaluate!(cache.value,b.system,x,cache.cfg)
  _fill_value!(v,cache.value)
end

function evaluate!(
  v::AbstractVector{T},
  b::FixedPolynomialBasis{P,T,G,T},
  x::P,
  cache::FixedScratchData{T}) where {P,G,T}

  fp.evaluate!(v,b.system,x,cache.cfg)
end

function gradient!(
  v::AbstractVector{G},
  b::FixedPolynomialBasis{P,V,G,T},
  x::P,
  cache::FixedScratchData{T}) where {P,V,G,T}

  fp.jacobian!(cache.jacob,b.system,x,cache.cfg)
  _fill_gradient!(v,cache.jacob,V)
end

function gradient!(
  v::AbstractVector{G},
  b::FixedPolynomialBasis{P,T,G,T},
  x::P,
  cache::FixedScratchData{T}) where {P,G,T}

  fp.jacobian!(cache.jacob,b.system,x,cache.cfg)
  _fill_gradient_scalar!(v,cache.jacob)
end

# Vectorized versions of evaluate! and gradient!

function evaluate!(
  v::AbstractMatrix{V},
  b::FixedPolynomialBasis{P,V,G,T},
  x::AbstractVector{P},
  cache::FixedScratchData{T,V}) where {P,V,G,T}

  vi = cache.v
  n = length(vi)
  for (i,xi) in enumerate(x)
    evaluate!(vi,b,xi,cache)
    for j in 1:n
      v[j,i] = vi[j]
    end
  end
end

function gradient!(
  v::AbstractMatrix{G},
  b::FixedPolynomialBasis{P,V,G,T},
  x::AbstractVector{P},
  cache::FixedScratchData{T,V,G}) where {P,V,G,T}

  vi = cache.g
  n = length(vi)
  for (i,xi) in enumerate(x)
    gradient!(vi,b,xi,cache)
    for j in 1:n
      v[j,i] = vi[j]
    end
  end
end

function _fill_value!(
  v::AbstractVector{V},
  s::AbstractVector{T}) where {V,T}

  m = zero(_mutable(V))
  k = 1
  for si in s
    for j in eachindex(m)
      m .= 0.0
      m[j] = si
      v[k] = m
      k += 1
    end
  end
end

function _fill_gradient!(
  v::AbstractVector{G},
  s::AbstractMatrix{T},
  ::Type{V}) where {T,V,G}

  m = zero(_mutable(G))
  w = zero(V)
  k = 1
  for i in 1:size(s,1)
    for l in CartesianIndices(w)
      m .= 0.0
      for j in 1:size(s,2)
        m[j,l] = s[i,j]
        v[k] = m
      end
      k += 1
    end 
  end
end

function _fill_gradient_scalar!(
  v::AbstractVector{G},
  s::AbstractMatrix{T}) where {T,G}

  m = zero(_mutable(G))
  k = 1
  for i in 1:size(s,1)
    m .= 0.0
    for j in 1:size(s,2)
      m[j] = s[i,j]
      v[k] = m
    end
    k += 1
  end
end
