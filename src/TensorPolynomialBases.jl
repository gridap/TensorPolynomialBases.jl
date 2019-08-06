module TensorPolynomialBases

using StaticArrays
using TensorValues
using Test

export ScratchData
export TensorPolynomialBasis
export MonomialBasis
export QGradMonomialBasis
export QCurlGradMonomialBasis
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

include("QCurlGradMonomialBases.jl")

end # module
