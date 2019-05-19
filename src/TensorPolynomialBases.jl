module TensorPolynomialBases

### To be removed

using TensorValues
using StaticArrays
import Base: convert
import Base: CartesianIndices
import Base: LinearIndices

function convert(::Type{<:MultiValue{S,T,N,L}},a::StaticArray{S,T,N}) where {S,T,N,L}
  MultiValue(a)
end

function CartesianIndices(a::MultiValue)
  CartesianIndices(a.array)
end

function LinearIndices(a::MultiValue)
  LinearIndices(a.array)
end

import Base: length
length(::Type{MultiValue{S,T,N,L}}) where {S,T,N,L} = L

import Base: isapprox
function isapprox(
  a::AbstractArray{<:MultiValue}, b::AbstractArray{<:MultiValue})
  if size(a) != size(b); return false; end
  for (ai,bi) in zip(a,b)
    if !(ai≈bi); return false; end
  end
  true
end

function convert(
  ::Type{<:MultiValue{S,T,N,L}},a::AbstractArray{T,N}) where {S,T,N,L}
  b = convert(SArray{S,T,N,L},a)
  MultiValue(b)
end

###

using StaticArrays
using TensorValues
using Test

export ScratchData
export TensorPolynomialBasis
export MonomialBasis
export test_polynomial_basis_without_gradient
export test_polynomial_basis
export gradient_type, value_type, point_type
export evaluate, gradient
export evaluate!, gradient!
import Base: length, ndims

include("Utils.jl")

include("Interfaces.jl")

include("MonomialBases.jl")

end # module
