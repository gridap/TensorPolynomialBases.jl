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

export FixedPolynomialBasis, ScratchData
export gradient_type, value_type, coeff_type
export evaluate, gradient
export evaluate!, gradient!
import Base: length, ndims
import FixedPolynomials: System

include("Utils.jl")

include("Interfaces.jl")

include("FixedPolynomialBases.jl")

end # module
