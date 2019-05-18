module TensorPolynomialBases

using TensorValues
using StaticArrays
import Base: convert
import Base: CartesianIndices
import Base: LinearIndices

function convert(::Type{<:MultiValue{S,T,N}},a::StaticArray{S,T,N}) where {S,T,N}
  MultiValue(a)
end

function CartesianIndices(a::MultiValue)
  CartesianIndices(a.array)
end

function LinearIndices(a::MultiValue)
  LinearIndices(a.array)
end

using DynamicPolynomials: @polyvar
import FixedPolynomials; const fp = FixedPolynomials
using StaticArrays
using TensorValues

export ScratchData
export TensorPolynomialBasis
export FixedPolynomialBasis
export QMonomialBasis
export gradient_type, value_type, point_type
export evaluate, gradient
export evaluate!, gradient!
import Base: length, ndims

include("Utils.jl")

include("Interfaces.jl")

include("FixedPolynomialBases.jl")

include("QMonomialBases.jl")

end # module
