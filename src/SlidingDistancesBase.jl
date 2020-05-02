module SlidingDistancesBase

using UnsafeArrays

export distance_profile, distance_profile!, lastlength, getwindow, floattype




floattype(T::Type{<:Integer}) = float(T)
floattype(T::Type{<:AbstractFloat}) = T
floattype(_) = Float64

Base.@propagate_inbounds Base.getindex(v::AbstractVector, ::typeof(!), i) = v[i]
Base.@propagate_inbounds Base.getindex(v::AbstractVector, ::typeof(!), i::AbstractRange) = uview(v, i)
Base.@propagate_inbounds Base.getindex(v::AbstractMatrix, ::typeof(!), i) = uview(v,:,i)
Base.@propagate_inbounds Base.getindex(v::AbstractArray{<:Any,3}, ::typeof(!), i) = uview(v,:,:,i)

Base.@propagate_inbounds Base.setindex!(v::AbstractVector, val, ::typeof(!), i) = v[i] = val
Base.@propagate_inbounds Base.setindex!(v::AbstractVector, val, ::typeof(!), i::AbstractRange) = v[i] .= val
Base.@propagate_inbounds Base.setindex!(v::AbstractMatrix, val, ::typeof(!), i) = v[:,i] .= val
Base.@propagate_inbounds Base.setindex!(v::AbstractArray{<:Any,3}, val, ::typeof(!), i) = v[:,:,i] .= val

lastlength(x) = size(x, ndims(x))

"""
    getwindow(T, m, i)

Extract a window of length `m` staring at `i`. The returned object will be an `UnsafeView` into `T`.
"""
getwindow(T,m,i) = T[!, (0:m-1) .+ i]


"""
    distance_profile(dist, Q, T; kwargs...)

Calculate the distance profile corresponding to sliding `Q` over `T` and measureing distances between time windows using `dist`

The output is a vector of length `length(T) - length(Q) + 1`.
If inputs are matrices or higher-dimensional arrays, time is considered to be the last axis, and windows on the form `T[:, t:t+lastlength(Q)-1]`, where `lastlength` measures the length along the last axis, are compared.

Keyword arguments are sent to `dist`, which is called like this `dist(Q,T; kwargs...)`. `dist` can be a function or a distance from Distances.jl.

`distance_profile!(D, dist, Q, T; kwargs...)` is an in-place version.
"""
function distance_profile(dist, Q::AbstractArray{S}, T::AbstractArray{S}; kwargs...) where S
    m = lastlength(Q)
    n = lastlength(T)
    n >= m || throw(ArgumentError("Q cannot be longer than T"))
    l = n-m+1
    D = map(1:l) do i
        dist(Q, getwindow(T, m, i); kwargs...)
    end
end

function distance_profile!(D::AbstractArray, dist, Q::AbstractArray{S}, T::AbstractArray{S}; kwargs...) where S
    m = lastlength(Q)
    n = lastlength(T)
    n >= m || throw(ArgumentError("Q cannot be longer than T"))
    l = n-m+1
    length(D) >= l || throw(ArgumentError("D is too short"))
    map!(D, 1:l) do i
        dist(Q, getwindow(T, m, i); kwargs...)
    end
end


end
