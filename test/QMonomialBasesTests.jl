module QMonomialBasesTests

using Test
using TensorPolynomialBases
using StaticArrays
using TensorValues

order = 1

P = SVector{2,Float64}
V = SVector{1,Float64}

basis = QMonomialBasis{P,V}(order)

cache = ScratchData(basis)

x = @SVector [1.0,2.0]

@show evaluate(basis,x,cache)




end # module QMonomialBasesTests
