# TensorPolynomialBases.jl

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
