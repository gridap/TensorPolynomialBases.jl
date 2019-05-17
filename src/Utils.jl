
function gradient_type(::Type{A},::Val{D}) where A<:StaticArray{S,T,N} where {S,T,N,D}
  SG = _gradient_size(Size(A),Val(D))
  TG = T
  NG = N+1
  LG = _gradient_length(Size(A),Val(D))
  SArray{SG,TG,NG,LG}
end

function gradient_type(::Type{A},::Val{D}) where A<:MultiValue{S,T,N,L} where {S,T,N,L,D}
  G = gradient_type(SArray{S,T,N,L},Val(D))
  _to_multivalue_type(G)
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

