
abstract type TensorPolynomialBasis{T,V,G} end

length(::TensorPolynomialBasis)::Int = @abstractmethod

ndims(::TensorPolynomialBasis)::Int = @abstractmethod

ScratchData(b::TensorPolynomialBasis) = @abstractmethod

function evaluate!(
  v::AbstractVector{V},
  b::TensorPolynomialBasis{T,V},
  x::AbstractVector{T},
  cache) where {T,V}
  @abstractmethod
end

function gradient!(
  v::AbstractVector{G},
  b::TensorPolynomialBasis{T,V,G},
  x::AbstractVector{T},
  cache) where {T,V,G}
  @abstractmethod
end

function evaluate!(
  v::AbstractMatrix{V},
  b::TensorPolynomialBasis{T,V},
  x::AbstractVector{<:AbstractVector{T}},
  cache) where {T,V}
  @abstractmethod
end

function gradient!(
  v::AbstractMatrix{G},
  b::TensorPolynomialBasis{T,V,G},
  x::AbstractVector{<:AbstractVector{T}},
  cache) where {T,V,G}
  @abstractmethod
end

gradient_type(::TensorPolynomialBasis{T,V,G}) where {T,V,G} = G

value_type(::TensorPolynomialBasis{T,V,G}) where {T,V,G} = V

coeff_type(::TensorPolynomialBasis{T,V,G}) where {T,V,G} = T

function evaluate(
  b::TensorPolynomialBasis{T,V},
  x::AbstractVector{T},
  cache) where {T,V}
  n = length(b)
  v = zeros(V,n)
  evaluate!(v,b,x,cache)
  v
end

function gradient(
  b::TensorPolynomialBasis{T,V,G},
  x::AbstractVector{T},
  cache) where {T,V,G}
  n = length(b)
  v = zeros(G,n)
  gradient!(v,b,x,cache)
  v
end

function evaluate(
  b::TensorPolynomialBasis{T,V},
  x::AbstractVector{<:AbstractVector{T}},
  cache) where {T,V}
  n = length(b)
  m = length(x)
  v = zeros(V,(n,m))
  evaluate!(v,b,x,cache)
  v
end

function gradient(
  b::TensorPolynomialBasis{T,V,G},
  x::AbstractVector{<:AbstractVector{T}},
  cache) where {T,V,G}
  n = length(b)
  m = length(x)
  v = zeros(G,(n,m))
  gradient!(v,b,x,cache)
  v
end


