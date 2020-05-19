
floattype(T::Type{<:Integer}) = float(T)
floattype(T::Type{<:AbstractFloat}) = T
floattype(_) = Float64

Base.@propagate_inbounds @inline Base.getindex(v::AbstractVector, ::typeof(!), i) = v[i]
Base.@propagate_inbounds @inline Base.getindex(v::AbstractVector, ::typeof(!), i::AbstractRange) = uview(v, i)
Base.@propagate_inbounds @inline Base.getindex(v::AbstractMatrix, ::typeof(!), i) = uview(v,:,i)
Base.@propagate_inbounds @inline Base.getindex(v::AbstractArray{<:Any,3}, ::typeof(!), i) = uview(v,:,:,i)

Base.@propagate_inbounds @inline Base.setindex!(v::AbstractVector, val, ::typeof(!), i) = v[i] = val
Base.@propagate_inbounds @inline Base.setindex!(v::AbstractVector, val, ::typeof(!), i::AbstractRange) = v[i] .= val
Base.@propagate_inbounds @inline Base.setindex!(v::AbstractMatrix, val, ::typeof(!), i) = v[:,i] .= val
Base.@propagate_inbounds @inline Base.setindex!(v::AbstractArray{<:Any,3}, val, ::typeof(!), i) = v[:,:,i] .= val

"Return the length of the last axis"
lastlength(x) = size(x, ndims(x))
actuallastlength(x) = lastlength(x)

"""
    getwindow(T, m, i)

Extract a window of length `m` staring at `i`. The returned object will be an `UnsafeView` into `T`.
"""
Base.@propagate_inbounds @inline getwindow(T,m,i) = T[!, (0:m-1) .+ i]
