module TensorPolynomialBasesTests

using TensorPolynomialBases
using Test

@testset "UtilsTests" begin
  include("UtilsTests.jl")
end

@testset "FixedPolynomialBasesTests" begin
  include("FixedPolynomialBasesTests.jl")
end

@testset "QMonomialBasesTests" begin
  include("QMonomialBasesTests.jl")
end

end # module TensorPolynomialBasesTests
