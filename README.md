# TensorPolynomialBases

[![Build Status](https://travis-ci.com/gridap/TensorPolynomialBases.jl.svg?branch=master)](https://travis-ci.com/gridap/TensorPolynomialBases.jl)
[![Codecov](https://codecov.io/gh/gridap/TensorPolynomialBases.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gridap/TensorPolynomialBases.jl)

The **TensorPolynomialBases** package provides a collection of different types representing tensor-valued multivariate polynomial bases. It provides a common interface, called `TensorPolynomialBasis`, and several concrete implementations. At the moment, only a concrete implementation, called `FixedPolynomialBasis`, which uses the [FixedPolynomials](https://github.com/JuliaAlgebra/FixedPolynomials.jl) package, is available. For representing the tensor values arising in the evaluation of tensor-valued polynomails, the user can either use the [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl) or the [TensorValues](https://github.com/gridap/TensorValues.jl) packages.

## Quick start

### Create a vector-valued monomial basis of P-polynomials in 2 variables

```julia
using TensorPolynomialBases
using StaticArrays

# Define a filter to select the monomials in the P-space
filter(e,order) = sum(e) <= order

order= 4
dim = 2
T = Float64 # type of the variables
V = SVector{3,Float64} # type of the value

basis = FixedPolynomialBasis{T,V}(filter,order,dim)

# Evaluation
x = rand(3)
cache = ScratchData(basis)
v = zeros(V,length(basis))
evaluate!(v,basis,x,cache) # No memory allocation here
@show v

# Evaluation of the gradient
G = gradient_type(basis)
# G == SMatrix{2,3,T,6}
v = zeros(G,length(basis))
gradient!(v,basis,x,cache) # No memory allocation here
@show v
```
