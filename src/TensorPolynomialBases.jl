module TensorPolynomialBases

using StaticArrays
using TensorValues
using Test

# Move to VectorValues (begin)
import Base: convert
function convert(::Type{<:MultiValue{S,T,N,L}},a::NTuple{L,T}) where {S,T,N,L}
  MultiValue(SArray{S,T}(a))
end
# Move to VectorValues (end)

export ScratchData
export TensorPolynomialBasis
export MonomialBasis
export QGradMonomialBasis
export test_polynomial_basis_without_gradient
export test_polynomial_basis
export gradient_type, value_type, point_type
export evaluate, gradient
export evaluate!, gradient!
import Base: length, ndims

include("Utils.jl")

include("Interfaces.jl")

include("MonomialBases.jl")

include("QGradMonomialBases.jl")

end # module
