# TensorPolynomialBases

[![Build Status](https://travis-ci.com/gridap/TensorPolynomialBases.jl.svg?branch=master)](https://travis-ci.com/gridap/TensorPolynomialBases.jl)
[![Codecov](https://codecov.io/gh/gridap/TensorPolynomialBases.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gridap/TensorPolynomialBases.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://gridap.github.io/TensorPolynomialBases.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://gridap.github.io/TensorPolynomialBases.jl/latest)

The **TensorPolynomialBases** package provides a collection of different types representing tensor-valued multivariate polynomial bases. It provides a common interface, called `TensorPolynomialBasis`, and several concrete implementations. At the moment, only a concrete implementation, called `MonomialBasis` is available, which implements a tensor-valued multivariate monomial basis. For representing the tensor values arising in the evaluation of tensor-valued polynomails, the user can either use the [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl) or the [TensorValues](https://github.com/gridap/TensorValues.jl) packages.

## Quick start

### Create a vector-valued monomial basis of P-polynomials in 2 variables

```julia
using TensorPolynomialBases
using StaticArrays

# Define a filter to select the monomials in the P-space
filter(e,order) = sum(e) <= order

order= 4
P = SVector{2,Float64} # type of the evaluation point
V = SVector{3,Float64} # type of the value

basis = MonomialBasis{P,V}(filter,order)

# Create scratch data that can be reused between evaluations
cache = ScratchData(basis)

# Evaluation point
x = @SVector rand(3)

# Evaluation
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

### Create a Tensor-valued monomial basis of the "serendipity" space in 3 variables (this time using the types of the TensorValues package)

```julia
using TensorValues

# Define the filter for the serendipity space
filter(e,order) = sum( ( i for i in e if i>1 ) ) <= order

order= 3
P = VectorValue{3,Float64} # type of the evaluation point
V = TensorValue{3,Float64,9} # type of the value (3x3 tensor)

basis = MonomialBasis{P,V}(filter,order)

# Create scratch data that can be reused between evaluations
cache = ScratchData(basis)

# Evaluation point
x = VectorValue(0.1,2.0,3.1)

# Evaluation
v = zeros(V,length(basis))
evaluate!(v,basis,x,cache) # No memory allocation here
@show v

```


