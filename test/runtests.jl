module TensorPolynomialBasesTests

using TensorPolynomialBases
using Test
import FixedPolynomials; const fp = FixedPolynomials
using StaticArrays

filter(e,O) = (sum(e) <= O)

order = 3
dim = 2
B = fp.System{Float64}(filter,order,dim)

cfg = fp.JacobianConfig(B)

x = [1.0,2.0]

@test fp.evaluate(B,x,cfg) ≈ [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 8.0]
@test fp.jacobian(B,x,cfg) ≈ [0.0 0.0; 1.0 0.0; 2.0 0.0; 3.0 0.0; 0.0 1.0;
                     2.0 1.0; 4.0 1.0; 0.0 4.0; 4.0 4.0; 0.0 12.0]

#order = 3
#basis = FixedPolynomialBasis{T,SVector{Z,T}}(filter,order)

@test gradient_type(SVector{3,Int},Val(2)) == SMatrix{2,3,Int,6}

@test gradient_type(SMatrix{3,4,Int},Val(2)) == SArray{Tuple{2,3,4},Int,3,24}


end # module TensorPolynomialBasesTests
