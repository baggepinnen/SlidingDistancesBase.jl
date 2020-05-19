

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

"$(SIGNATURES)"
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

"$(SIGNATURES)"
function distance_profile!(D::AbstractVector{S},::ZEuclidean, QT::AbstractVector{S}, μ, σ, m::Int, i::Int) where S <: Number
    @assert i <= length(D)
    @avx for j = eachindex(D)
        frac = (QT[j] - m*μ[i]*μ[j]) / (m*σ[i]*σ[j])
        D[j] = sqrt(max(2m*(1-frac), 0))
    end
    D[i] = typemax(eltype(D))
    D
end

"$(SIGNATURES)"
distance_profile(
    ::ZEuclidean,
    QT::AbstractVector{S},
    μ::AbstractVector{S},
    σ::AbstractVector{S},
    m::Int,
) where {S<:Number} = distance_profile!(similar(μ), ZEuclidean(), QT, μ, σ, m, 1)

"""
    $(SIGNATURES)

Accepts precomputed sliding mean and std of the input arrays. `QT` is the windowed dot product and `i` is the index into `T` (the longer time series).
"""
function distance_profile!(D::AbstractVector{S},::ZEuclidean, QT::AbstractVector{S}, μA, σA, μT, σT, m::Int, i::Int) where S <: Number
    @assert i <= length(μA)
    @avx for j = eachindex(D,QT,μT,σT)
        frac = (QT[j] - m*μA[i]*μT[j]) / (m*σA[i]*σT[j])
        D[j] = sqrt(max(2m*(1-frac), 0))
    end
    D
end

"$(SIGNATURES)"
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
    μ, σ = sliding_mean_std(T, m)
    QT = window_dot(znorm(Q), T) # TODO: allocates a new znorm(Q) each time
    @avx for j in eachindex(D,QT,σ)
        frac = QT[j] / (m * σ[j])
        D[j] = sqrt(max(2m * (1 - frac), 0))
    end
    D
end

"$(SIGNATURES)"
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

"""
    $(SIGNATURES)

return vectors with mean and std of sliding windows of length `m`
"""
function sliding_mean_std(x::AbstractArray{T}, m) where T
    @assert length(x) >= m
    n = length(x)-m+1
    s = ss = zero(float(T))
    μ = Vector{float(T)}(undef, n)
    σ = Vector{float(T)}(undef, n)
    @avx for i = 1:m
        s  += x[i]
        ss += x[i]^2
    end
    μ[1] = s/m
    σ[1] = sqrt(max(ss/m - μ[1]^2, 0))
    @fastmath @inbounds for i = 1:n-1 # fastmath making it more accurate here as well, but not faster
        s -= x[i]
        ss -= x[i]^2
        s += x[i+m]
        ss += x[i+m]^2
        μ[i+1] = s/m
        σ[i+1] = sqrt(max(ss/m - μ[i+1]^2, 0))
    end
    μ,σ
end

"""
    $(SIGNATURES)

return mean of sliding windows of length `m`. Operates in-place and stores result in first argument
"""
function sliding_mean!(μ,x::AbstractArray{T}, m) where T
    @assert length(x) >= m
    n = length(x)-m+1
    s = zero(T)
    @avx for i = 1:m
        s += x[i]
    end
    μ[1] = s/m
    @fastmath @inbounds for i = 1:n-1 # fastmath making it more accurate here as well, but not faster
        s -= x[i]
        s += x[i+m]
        μ[i+1] = s/m
    end
    μ
end
