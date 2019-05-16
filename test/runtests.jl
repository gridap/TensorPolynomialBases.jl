module TensorPolynomialBasesTests

using TensorPolynomialBases
using Test

@testset "UtilsTests" begin
  include("UtilsTests.jl")
end

@testset "FixedPolynomialBasesTests" begin
  include("FixedPolynomialBasesTests.jl")
end

end # module TensorPolynomialBasesTests
