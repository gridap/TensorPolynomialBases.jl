module UtilsTests

using TensorPolynomialBases: _gradient_type
using Test
using StaticArrays
using TensorValues

@test _gradient_type(SVector{3,Int},SVector{2,Int}) == SMatrix{2,3,Int,6}

@test _gradient_type(SMatrix{3,4,Int},SVector{2,Int}) == SArray{Tuple{2,3,4},Int,3,24}

@test _gradient_type(VectorValue{3,Int},VectorValue{2,Int}) == MultiValue{Tuple{2,3},Int,2,6}

@test _gradient_type(TensorValue{3,Int,9},VectorValue{2,Int}) == MultiValue{Tuple{2,3,3},Int,3,18}

end # module UtilsTests
