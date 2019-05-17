
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

function evaluate!(
  v::AbstractMatrix{V},
  b::TensorPolynomialBasis{P,V},
  x::AbstractVector{P},
  cache) where {P,V}
  @abstractmethod
end

function gradient!(
  v::AbstractMatrix{G},
  b::TensorPolynomialBasis{P,V,G},
  x::AbstractVector{P},
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

function evaluate(
  b::TensorPolynomialBasis{P,V}, x::AbstractVector{P}, cache) where {P,V}
  n = length(b)
  m = length(x)
  v = zeros(V,(n,m))
  evaluate!(v,b,x,cache)
  v
end

function gradient(
  b::TensorPolynomialBasis{P,V,G}, x::AbstractVector{P}, cache) where {P,V,G}
  n = length(b)
  m = length(x)
  v = zeros(G,(n,m))
  gradient!(v,b,x,cache)
  v
end

