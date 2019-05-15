module TensorPolynomialBasesTests

using TensorPolynomialBases
using Test
using FixedPolynomials

filter(e,O) = (sum(e) <= O)

order = 3
dim = 2
B = System{Float64}(filter,order,dim)

cfg = JacobianConfig(B)

x = [1.0,2.0]

@test evaluate(B,x,cfg) ≈ [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 8.0]
@test jacobian(B,x,cfg) ≈ [0.0 0.0; 1.0 0.0; 2.0 0.0; 3.0 0.0; 0.0 1.0;
                     2.0 1.0; 4.0 1.0; 0.0 4.0; 4.0 4.0; 0.0 12.0]

end # module TensorPolynomialBasesTests
