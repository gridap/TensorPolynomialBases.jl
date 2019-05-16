module UtilsTests

using TensorPolynomialBases
using Test
using StaticArrays

@test gradient_type(SVector{3,Int},Val(2)) == SMatrix{2,3,Int,6}

@test gradient_type(SMatrix{3,4,Int},Val(2)) == SArray{Tuple{2,3,4},Int,3,24}

end # module UtilsTests
