module TensorPolynomialBases

using DynamicPolynomials: @polyvar
import FixedPolynomials; const fp = FixedPolynomials
using StaticArrays

export FixedPolynomialBasis, ScratchData
export gradient_type, value_type, coeff_type
export evaluate!, gradient!
import Base: length, ndims
import FixedPolynomials: System

include("Utils.jl")

include("FixedPolynomialBases.jl")

end # module
