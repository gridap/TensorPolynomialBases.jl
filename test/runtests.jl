module TensorPolynomialBasesTests

using TensorPolynomialBases
using Test

@testset "UtilsTests" begin
  include("UtilsTests.jl")
end

@testset "InterfacesTests" begin
  include("InterfacesTests.jl")
end

@testset "MonomialBasesTests" begin
  include("MonomialBasesTests.jl")
end

end # module TensorPolynomialBasesTests
