module TensorPolynomialBases

using FixedPolynomials
using DynamicPolynomials: @polyvar

import FixedPolynomials: System

function (::Type{System{T}})(filter::Function, order::Int, dim::Int) where T
  B = _setup_system(filter,order,dim,T)
end

function _setup_system(filter,O,D,T)
  @polyvar x[1:D]
  o = CartesianIndex(ones(Int,D)...)
  cs = CartesianIndices( tuple(fill(O+1,D)...) )
  A = Polynomial{T}[]
  for c in cs
    e = Tuple(c-o)
    if filter(e,O)
      f = 1
      for (xi,p) in zip(x,e)
        f *= xi^p
      end
      poly = Polynomial{Float64}(f)
      push!(A,f)
    end
  end
  System(A)
end

end # module
