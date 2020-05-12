module SlidingDistancesBase

using Statistics, LinearAlgebra

using UnsafeArrays
using Distances
export evaluate

using LoopVectorization
using DSP

export distance_profile, distance_profile!, lastlength, getwindow, floattype, znorm, ZEuclidean, meanstd, running_mean_std, running_mean!


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

"""
    getwindow(T, m, i)

Extract a window of length `m` staring at `i`. The returned object will be an `UnsafeView` into `T`.
"""
Base.@propagate_inbounds @inline getwindow(T,m,i) = T[!, (0:m-1) .+ i]


"""
    distance_profile(dist, Q, T; kwargs...)

Calculate the distance profile corresponding to sliding `Q` over `T` and measuring distances between time windows using `dist`

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


function znorm(x::AbstractVector)
    x = x .- mean(x)
    x ./= std(x, mean=0, corrected=false)
end

struct ZEuclidean <: Distances.Metric end

function meanstd(x::AbstractArray{T}) where T
    s = ss = zero(T)
    n = length(x)
    @avx for i in eachindex(x)
        s  += x[i]
        ss += x[i]^2
    end
    m   = s/n
    sig = √(ss/n - m^2)
    m, sig
end


function Distances.evaluate(d::ZEuclidean, x::AbstractArray{T}, y::AbstractArray{T}) where T
    mx,sx = meanstd(x)
    my,sy = meanstd(y)
    s = zero(T)
    @avx for i in eachindex(x,y)
        s += ((x[i]-mx)/sx - (y[i]-my)/sy)^2
    end
    √(s)
end

(d::ZEuclidean)(x, y) = evaluate(d, x, y)



function distance_profile!(D::AbstractVector{S},::ZEuclidean, QT::AbstractVector{S}, μ, σ, m::Int, i::Int) where S <: Number
    @assert i <= length(D)
    @avx for j = eachindex(D)
        frac = (QT[j] - m*μ[i]*μ[j]) / (m*σ[i]*σ[j])
        D[j] = sqrt(max(2m*(1-frac), 0))
    end
    D[i] = typemax(eltype(D))
    D
end

distance_profile(
    ::ZEuclidean,
    QT::AbstractVector{S},
    μ::AbstractVector{S},
    σ::AbstractVector{S},
    m::Int,
) where {S<:Number} = distance_profile!(similar(μ), ZEuclidean(), QT, μ, σ, m, 1)


function distance_profile!(D::AbstractVector{S},::ZEuclidean, QT::AbstractVector{S}, μA, σA, μT, σT, m::Int, i::Int) where S <: Number
    @assert i <= length(μA)
    @avx for j = eachindex(D,QT,μT,σT)
        frac = (QT[j] - m*μA[i]*μT[j]) / (m*σA[i]*σT[j])
        D[j] = sqrt(max(2m*(1-frac), 0))
    end
    D
end

distance_profile(::ZEuclidean, QT::AbstractVector{S}, μA, σA, μT, σT, m::Int) where {S<:Number} =
    distance_profile!(similar(μT), ZEuclidean(), QT, μA, σA, μT, σT, m, 1)


"""
    distance_profile(::ZEuclidean, Q, T)

Compute the z-normalized Euclidean distance profile corresponding to sliding `Q` over `T`
"""
function distance_profile!(
    D::AbstractVector{S},
    ::ZEuclidean,
    Q::AbstractVector{S},
    T::AbstractVector{S},
) where {S<:Number}
    m = length(Q)
    μ, σ = running_mean_std(T, m)
    QT = window_dot(znorm(Q), T)
    @avx for j in eachindex(D)
        frac = QT[j] / (m * σ[j])
        D[j] = sqrt(max(2m * (1 - frac), 0))
    end
    D
end
distance_profile(::ZEuclidean, Q::AbstractArray{S}, T::AbstractArray{S}) where {S} =
    distance_profile!(similar(T, length(T) - length(Q) + 1), ZEuclidean(), Q, T)

"""
The dot product between query Q and all subsequences of the same length as Q in time series T
"""
function window_dot(Q, T)
    n   = length(T)
    m   = length(Q)
    QT  = conv(reverse(Q), T)
    return QT[m:n]
end


struct ConvPlan{UT,P,CT}
    padded::UT
    plan::P
    uf::CT
    vf::CT
    raw_out::UT
    out::UT
    nfft::Int
end

function ConvPlan(u,v)
    if length(v) > length(u)
        u,v = v,u
    end
    su = size(u)
    sv = size(v)

    outsize = su .+ sv .- 1
    out = DSP._conv_similar(u, v, outsize)
    nfft = DSP.nextfastfft(outsize)[1]
    padded = DSP._zeropad(u, (nfft,))
    plan = DSP.plan_rfft(padded)

    uf = Vector{Complex{eltype(u)}}(undef, nfft÷2+1)
    vf = Vector{Complex{eltype(v)}}(undef, nfft÷2+1)
    raw_out = similar(u)
    out = similar(u, outsize)
    ConvPlan(padded, plan, uf, vf, raw_out, out, nfft)
end

function conv!(p::ConvPlan, u, v)
    if length(v) > length(u)
        u,v = v,u
    end
    su = size(u)
    sv = size(v)
    outsize = su .+ sv .- 1

    p.padded .= 0
    copyto!(p.padded, 1, u, 1, length(u))
    mul!(p.uf, p.plan, p.padded)
    DSP._zeropad!(p.padded, v)
    mul!(p.vf, p.plan, p.padded)
    p.uf .*= p.vf
    raw_out = DSP.irfft(p.uf, p.nfft)
    # copyto!(p.out,
    #         CartesianIndices(p.out),
    #         p.raw_out,
    #         CartesianIndices(UnitRange.(1, outsize)))
end


function window_dot!(p::ConvPlan, Q, T)
    n   = length(T)
    m   = length(Q)
    QT  = conv!(p, reverse(Q), T)
    return QT[m:n]
end

function running_mean_std(x::AbstractArray{T}, m) where T
    @assert length(x) >= m
    n = length(x)-m+1
    s = ss = zero(T)
    μ = similar(x, n)
    σ = similar(x, n)
    @avx for i = 1:m
        s  += x[i]
        ss += x[i]^2
    end
    μ[1] = s/m
    σ[1] = sqrt(ss/m - μ[1]^2)
    @fastmath @inbounds for i = 1:n-1 # fastmath making it more accurate here as well, but not faster
        s -= x[i]
        ss -= x[i]^2
        s += x[i+m]
        ss += x[i+m]^2
        μ[i+1] = s/m
        σ[i+1] = sqrt(ss/m - μ[i+1]^2)
    end
    μ,σ
end

function running_mean!(μ,x::AbstractArray{T}, m) where T
    @assert length(x) >= m
    n = length(x)-m+1
    s = zero(T)
    @avx for i = 1:m
        s  += x[i]
    end
    μ[1] = s/m
    @fastmath @inbounds for i = 1:n-1 # fastmath making it more accurate here as well, but not faster
        s -= x[i]
        s += x[i+m]
        μ[i+1] = s/m
    end
    μ
end


end
