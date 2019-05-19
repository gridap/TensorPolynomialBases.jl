
const PointType{D} = Union{VectorValue{D}, SVector{D}}

abstract type TensorPolynomialBasis{P,V,G} end

length(::TensorPolynomialBasis)::Int = @abstractmethod

ndims(::TensorPolynomialBasis)::Int = @abstractmethod

ScratchData(b::TensorPolynomialBasis) = @abstractmethod

function evaluate!(
  v::AbstractVector{V},
  b::TensorPolynomialBasis{P,V},
  x::P,
  cache) where {P,V}
  @abstractmethod
end

function gradient!(
  v::AbstractVector{G},
  b::TensorPolynomialBasis{P,V,G},
  x::P,
  cache) where {P,V,G}
  @abstractmethod
end

gradient_type(::TensorPolynomialBasis{P,V,G}) where {P,V,G} = G

value_type(::TensorPolynomialBasis{P,V,G}) where {P,V,G} = V

point_type(::TensorPolynomialBasis{P}) where P = P

function evaluate(
  b::TensorPolynomialBasis{P,V}, x::P, cache) where {P,V}
  n = length(b)
  v = zeros(V,n)
  evaluate!(v,b,x,cache)
  v
end

function gradient(
  b::TensorPolynomialBasis{P,V,G}, x::P, cache) where {P,V,G}
  n = length(b)
  v = zeros(G,n)
  gradient!(v,b,x,cache)
  v
end

function evaluate(b::TensorPolynomialBasis{P,V}, x::P) where {P,V}
  cache = ScratchData(b)
  evaluate(b,x,cache)
end

function gradient(b::TensorPolynomialBasis{P,V,G}, x::P) where {P,V,G}
  cache = ScratchData(b)
  gradient(b,x,cache)
end

function test_polynomial_basis_without_gradient(
  basis::TensorPolynomialBasis{P,V},
  x::P,
  v::AbstractVector{V}) where {P,V}

  @test length(basis) == length(v)

  @test ndims(basis) == _length(P)

  cache = ScratchData(basis)

  v2 = evaluate(basis,x,cache)

  @test v ≈ v2

end

function test_polynomial_basis(
  basis::TensorPolynomialBasis{P,V,G},
  x::P,
  v::AbstractVector{V},
  g::AbstractVector{G}) where {P,V,G}

  test_polynomial_basis_without_gradient(basis,x,v)

  cache = ScratchData(basis)

  g2 = gradient(basis,x,cache)

  @test g ≈ g2

end
