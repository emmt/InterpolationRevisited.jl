struct BSplinePoles{D,T<:AbstractFloat,N}
    poles::NTuple{N,T}
    BSplinePoles{D,T}(poles::Vararg{Real,N}) where {D,T<:AbstractFloat,N} =
        new{D,T,N}(poles)
end
