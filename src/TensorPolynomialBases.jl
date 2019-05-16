module TensorPolynomialBases

using DynamicPolynomials: @polyvar
import FixedPolynomials; const fp = FixedPolynomials
using StaticArrays

export FixedPolynomialBasis, ScratchData
export gradient_type, value_type, coeff_type
export evaluate!, gradient!
import Base: length, ndims

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

"""
Scratch data associated with FixedPolynomialBasis
"""
struct FixedScratchData{T,C<:fp.JacobianConfig}
  cfg::C
  value::Vector{T}
  jacob::Matrix{T}
end

function FixedScratchData(system::fp.System{T}) where T
  cfg = fp.JacobianConfig(system)
  n = length(system)
  d = fp.nvariables(system)
  value = zeros(T,n)
  jacob = zeros(T,(n,d))
  FixedScratchData(cfg,value,jacob)
end

"""
Implementation of a tensor-valued multivariate polynomial basis
using the functionality given by the package FixedPolynomials
"""
struct FixedPolynomialBasis{T,V,G}
  system::fp.System{T}
end

"""
Generates a `FixedPolynomialBasis` object, whose coefficients
are of type `T` and the value of typ `V`
"""
function (::Type{FixedPolynomialBasis{T,V}})(
  filter::Function,order::Int,dim::Int) where {T,V}
  system = fp.System{T}(filter,order,dim)
  G = gradient_type(V,Val(dim))
  FixedPolynomialBasis{T,V,G}(system)
end

function length(b::FixedPolynomialBasis{T,V}) where {T,V}
  length(zero(V)) * length(b.system)
end

ndims(b::FixedPolynomialBasis) = nvariables(b.system)

function ScratchData(b::FixedPolynomialBasis)
  FixedScratchData(b.system)
end

function evaluate!(
  v::AbstractVector{V},
  b::FixedPolynomialBasis{T,V},
  x::AbstractVector{T},
  cache::FixedScratchData{T}) where {T,V}

  fp.evaluate!(cache.value,b.system,x,cache.cfg)
  _fill_value!(v,cache.value)
end

function gradient!(
  v::AbstractVector{G},
  b::FixedPolynomialBasis{T,V,G},
  x::AbstractVector{T},
  cache::FixedScratchData{T}) where {T,V,G}

  fp.jacobian!(cache.jacob,b.system,x,cache.cfg)
  _fill_gradient!(v,cache.jacob,V)
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

function gradient_type(::Type{A},::Val{D}) where A<:StaticArray{S,T,N} where {S,T,N,D}
  SG = _gradient_size(Size(A),Val(D))
  TG = T
  NG = N+1
  LG = _gradient_length(Size(A),Val(D))
  SArray{SG,TG,NG,LG}
end

gradient_type(::FixedPolynomialBasis{T,V,G}) where {T,V,G} = G

value_type(::FixedPolynomialBasis{T,V,G}) where {T,V,G} = V

coeff_type(::FixedPolynomialBasis{T,V,G}) where {T,V,G} = T

@generated function _gradient_size(::Size{B},::Val{D}) where {B,D}
  str = join(["$b," for b in B])
  Meta.parse("Tuple{$D,$str}")
end

function _gradient_length(::Size{B},::Val{D}) where {B,D}
  prod((D,B...))
end

_mutable(::Type{SArray{S,T,N,L}}) where {S,T,N,L} = MArray{S,T,N,L}

end # module
