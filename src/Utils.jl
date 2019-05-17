
function _gradient_type(
  ::Type{A},::Type{<:SVector{D}}) where A<:StaticArray{S,T,N} where {S,T,N,D}
  SG = _gradient_size(Size(A),Val(D))
  TG = T
  NG = N+1
  LG = _gradient_length(Size(A),Val(D))
  SArray{SG,TG,NG,LG}
end

function _gradient_type(
  ::Type{A},::Type{<:VectorValue{D}}) where A<:MultiValue{S,T,N,L} where {S,T,N,L,D}
  G = _gradient_type(SArray{S,T,N,L},SVector{D,T})
  _to_multivalue_type(G)
end

function _gradient_type(::Type{T},::Type{<:VectorValue{D}}) where {T<:Real,D}
  VectorValue{D,T}
end

function _gradient_type(::Type{T},::Type{<:SVector{D}}) where {T<:Real,D}
  SVector{D,T}
end

function _to_multivalue_type(::Type{SArray{S,T,N,L}}) where {S,T,N,L}
  MultiValue{S,T,N,L}
end

@generated function _gradient_size(::Size{B},::Val{D}) where {B,D}
  str = join(["$b," for b in B])
  Meta.parse("Tuple{$D,$str}")
end

function _gradient_length(::Size{B},::Val{D}) where {B,D}
  prod((D,B...))
end

_mutable(::Type{SArray{S,T,N,L}}) where {S,T,N,L} = MArray{S,T,N,L}

_mutable(::Type{MultiValue{S,T,N,L}}) where {S,T,N,L} = MArray{S,T,N,L}

macro abstractmethod()
  quote
    error("This function belongs to an interface definition and cannot be used.")
  end
end

