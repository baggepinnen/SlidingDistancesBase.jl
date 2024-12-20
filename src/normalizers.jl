"Z-normalize the input array"
function znorm(x::AbstractVector)
    x = x .- mean(x)
    x ./= std(x, mean=0, corrected=false)
end

function znorm!(x::AbstractVector)
    x .= x .- mean(x)
    x ./= std(x, mean=0, corrected=false)
end

"""
    meanstd(x::AbstractArray)

Return mean and std of `x`. Is faster but less accurate than built-in functions.
"""
function meanstd(x::AbstractArray{T}) where T
    s = ss = zero(float(T))
    n = length(x)
    @avx for i in eachindex(x)
        s  += x[i]
        ss += x[i]^2
    end
    m   = s/n
    sig = √(ss/n - m^2)
    m, sig
end


import Base: @propagate_inbounds, @boundscheck, getindex
import LinearAlgebra: normalize

abstract type AbstractNormalizer{T,N} <: AbstractArray{T,N} end

advance!(x) = 0 # Catches everything that's not a normalizer

setup_normalizer(n::Type{Nothing}, q, y) = q, y

normalize(::Type{Nothing}, q::Real) = q

(z::AbstractNormalizer)(q) = normalize(z, q)
(::Type{N})(q) where N <: AbstractNormalizer = normalize(N, q)


abstract type AbstractZNormalizer{T,N} <: AbstractNormalizer{T,N} end

"""
    ZNormalizer{T} <: AbstractNormalizer{T, 1}

Utility object that normalizes an input array on the fly.
Works like a vector, index into window with `z[i]`, index into normalized window with `z[!,i]`.

# Arguments:
- `x::Vector{T}`: The vecetor to operate on
- `n::Int`: The lenght of a window
- `μ::T`: mean over window
- `σ::T`: std over window
- `s::T`: sum over window
- `ss::T`: sum^2 over window
- `i::Int`: the current index
- `buffer::Vector{T}`: stores calculated normalized values, do not access this, it might not be complete
- `bufi::Int`: index of last normalized value
"""
mutable struct ZNormalizer{T,N} <: AbstractZNormalizer{T,N}
    x::Array{T,N}
    n::Int
    μ::T
    σ::T
    s::T
    ss::T
    i::Int
    buffer::Array{T,N}
    bufi::Int
end

function ZNormalizer(x::AbstractVector{T}, n) where T
    @assert length(x) >= n
    s = ss = zero(T)
    @avx for i in 1:n
        s += x[i]
        ss += x[i]^2
    end
    μ = s/n
    σ = sqrt(max(0, ss/n - μ^2))
    buffer = similar(x, n)
    ZNormalizer(x, n, μ, σ, s, ss, 0, buffer, 0)
end

function ZNormalizer(x::AbstractMatrix{T}, n) where T
    @assert length(x) >= n
    m = size(x,1)*n
    s = ss = zero(T)
    for i in 1:n
        s += sum(x[!,i])
        ss += sum(abs2, x[!,i])
    end
    μ = s/m
    σ = sqrt(max(0, ss/m - μ^2))
    buffer = similar(x, size(x,1), n)
    ZNormalizer(x, n, μ, σ, s, ss, 0, buffer, 0)
end


function normalize(::Type{ZNormalizer}, q::AbstractArray)
    q = q .- mean(q)
    q ./= std(q, corrected=false, mean=0)
end

setup_normalizer(z::Type{ZNormalizer}, q::AbstractVecOrMat, y::AbstractVecOrMat) = normalize(z, q), ZNormalizer(y, lastlength(q))

@propagate_inbounds function advance!(z::ZNormalizer{T,1}) where T
    if z.i == 0
        return z.i = 1
    end
    @boundscheck if z.i + z.n > length(z.x)
        return z.i += 1
    end
    z.bufi = 0

    # Remove old point
    x = z.x[z.i]
    z.s -= x
    z.ss -= x^2

    # Add new point
    x = z.x[z.i+z.n]
    z.s += x
    z.ss += x^2
    z.μ = z.s/z.n
    z.σ = sqrt(max(z.ss/z.n - z.μ^2, 0))
    z.i += 1
end

@propagate_inbounds function advance!(z::ZNormalizer{T,2}) where T
    if z.i == 0
        return z.i = 1
    end
    @boundscheck if z.i + z.n > lastlength(z.x)
        return z.i += 1
    end
    z.bufi = 0

    # Remove old point
    x = z.x[!, z.i]
    z.s -= sum(x)
    z.ss -= sum(abs2, x)

    # Add new point
    x = z.x[!, z.i+z.n]
    z.s += sum(x)
    z.ss += sum(abs2, x)
    m = length(z)
    z.μ = z.s/m
    z.σ = sqrt(max(z.ss/m - z.μ^2, 0))
    z.i += 1
end


@inline @propagate_inbounds function getindex(z::ZNormalizer{<:Any, 1}, i)
    @boundscheck 1 <= i <= z.n || throw(BoundsError(z,i))
    xi = i+z.i-1
    @boundscheck xi <= length(z.x) || throw(BoundsError(z,i))
    z.x[xi]
end

@inline @propagate_inbounds function getindex(z::ZNormalizer{<:Any, 1}, ::typeof(!), i::Int, inorder = i == z.bufi + 1)
    y = (z[i]-z.μ) / z.σ
    if inorder
        z.bufi = i
        z.buffer[i] = y
    end
    y
end

@inline @propagate_inbounds function getindex(z::ZNormalizer{<:Any, 1}, ::typeof(!), i::AbstractRange)
    @boundscheck (i[1] == z.i && length(i) == z.n) || throw(ArgumentError("ZNormalizers can only be indexed by ranges corresponding to their current state. Got range $i but state was $(z.i) corresponding to range $(z.i):$(z.i+z.n-1)"))
    z
end

@inline @propagate_inbounds function getindex(z::ZNormalizer{T,2}, ::typeof(!), i::Int, inorder = i == z.bufi + 1) where T
    j = inorder ? i : z.n
    xj = z.i + i - 1
    μ,σ = z.μ, z.σ + eps(T)
    @avx for k = 1:size(z.x, 1)
        z.buffer[k, j] = (z.x[k, xj] - μ) / σ
    end
    if inorder
        z.bufi = i
    end
    z.buffer[!, j]
end

Statistics.mean(z::AbstractZNormalizer) = z.μ
Statistics.std(z::AbstractZNormalizer) = z.σ

SlidingDistancesBase.lastlength(z::AbstractZNormalizer) = z.n
Base.length(z::AbstractZNormalizer) = length(z.buffer)
Base.size(z::ZNormalizer, args...) = size(z.buffer, args...)
actuallastlength(x::AbstractZNormalizer) = lastlength(x.x)




# Multi dim ==================================================================================
mutable struct DiagonalZNormalizer{T} <: AbstractZNormalizer{T,2}
    x::Array{T,2}
    n::Int
    μ::Array{T,1}
    σ::Array{T,1}
    s::Array{T,1}
    ss::Array{T,1}
    i::Int
    buffer::Array{T,2}
    bufi::Int
end



function DiagonalZNormalizer(x::AbstractArray{T,2}, n) where T
    @assert length(x) >= n
    s  = zeros(T, size(x,1))
    ss = zeros(T, size(x,1))
    @inbounds @simd for i in 1:n
        s  .+= x[!, i]
        ss .+= x[!, i].^2
    end
    μ = s./n
    σ = sqrt.(max.(ss./n .- μ.^2, 0))
    buffer = similar(x, size(x,1), n)
    DiagonalZNormalizer(x, n, μ, σ, s, ss, 0, buffer, 0)
end

function normalize(::Type{DiagonalZNormalizer}, q::AbstractMatrix)
    q = q .- mean(q, dims=2) # TODO: this will cause a ton of allocations
    q ./= (std(q, dims=2, corrected=false) .+ eps(eltype(q)))
end

setup_normalizer(z::Type{DiagonalZNormalizer}, q, y) = normalize(z, q), DiagonalZNormalizer(y, lastlength(q))


@propagate_inbounds function advance!(z::DiagonalZNormalizer{T}) where T

    if z.i == 0
        return z.i = 1
    end
    @boundscheck if z.i + z.n > length(z.x)
        return z.i += 1
    end
    z.bufi = 0

    # Remove old point
    x = z.x[!, z.i]
    @avx z.s .-= x
    @avx z.ss .-= x.^2

    # Add new point
    x = z.x[!, z.i+z.n]
    @avx z.s .+= x
    @avx z.ss .+= x.^2
    @avx z.μ .= z.s./z.n
    @avx z.σ .= sqrt.(z.ss./z.n .- z.μ.^2)
    z.i += 1
end



@inline @propagate_inbounds function getindex(z::AbstractNormalizer{<:Any, 2}, i::Union{Number, AbstractRange}, j)
    @boundscheck 1 <= j <= z.n || throw(BoundsError(z,j))
    @boundscheck 1 <= 1 <= size(z.x, 1) || throw(BoundsError(z,i))
    xj = j+z.i-1
    @boundscheck xj <= lastlength(z.x) || throw(BoundsError(z,j))
    z.x[i, xj]
end

@inline @propagate_inbounds function getindex(z::DiagonalZNormalizer{T}, ::typeof(!), i::Int, inorder = i == z.bufi + 1) where T
    j = inorder ? i : z.n
    xj = z.i + i - 1
    @avx for k = 1:size(z.x, 1)
        z.buffer[k, j] = (z.x[k, xj] - z.μ[k]) / (z.σ[k] + eps(T))
    end
    if inorder
        z.bufi = i
    end
    z.buffer[!, j]
end

@inline @propagate_inbounds function getindex(z::AbstractNormalizer{<:Any, 2}, ::typeof(!), i::AbstractRange)
    @boundscheck (i[1] == z.i && length(i) == z.n) || throw(ArgumentError("Normalizers can only be indexed by ranges corresponding to their current state. Got range $i but state was $(z.i) corresponding to range $(z.i):$(z.i+z.n-1)"))
    z
end

Base.Matrix(z::AbstractNormalizer{<:Any, 2}) = z.x[:,z.i:z.i+z.n-1]

Base.length(z::DiagonalZNormalizer) = size(z.x,1) * z.n
Base.size(z::AbstractNormalizer{<:Any, 2}) = (size(z.x,1), z.n)





# Norm normalizer ==================================================================================

abstract type AbstractNormNormalizer{T,N} <: AbstractNormalizer{T,N} end

mutable struct NormNormalizer{T,N} <: AbstractNormNormalizer{T,N}
    x::Array{T,N}
    n::Int
    σ::T
    ss::T
    i::Int
    buffer::Array{T,N}
    bufi::Int
end

mutable struct SqNormNormalizer{T,N} <: AbstractNormNormalizer{T,N}
    x::Array{T,N}
    n::Int
    σ::T
    ss::T
    i::Int
    buffer::Array{T,N}
    bufi::Int
end


function NormNormalizer(x::AbstractArray{T,N}, n) where {T,N}
    @assert length(x) >= n
    ss = zero(T)
    @inbounds for i in 1:n
        ss += sum(abs2, x[!, i])
    end
    σ = sqrt(ss)
    if N == 1
        buffer = similar(x, n)
    else
        buffer = similar(x, size(x,1), n)
    end
    NormNormalizer(x, n, σ, ss, 0, buffer, 0)
end

function SqNormNormalizer(x::AbstractArray{T,N}, n) where {T,N}
    @assert length(x) >= n
    ss = zero(T)
    @inbounds for i in 1:n
        ss += sum(abs2, x[!, i])
    end
    σ = ss
    if N == 1
        buffer = similar(x, n)
    else
        buffer = similar(x, size(x,1), n)
    end
    SqNormNormalizer(x, n, σ, ss, 0, buffer, 0)
end

function normalize(::Type{N}, q::AbstractArray) where N <: AbstractNormNormalizer
    power = N == NormNormalizer ? 1 : 2
    q ./ (norm(q)^power .+ eps(eltype(q)))
end


setup_normalizer(z::Type{N}, q, y) where N <: AbstractNormNormalizer = normalize(N, q), N(y, lastlength(q))


@propagate_inbounds function advance!(z::AbstractNormNormalizer{T}) where T

    if z.i == 0
        return z.i = 1
    end
    @boundscheck if z.i + z.n > length(z.x)
        return z.i += 1
    end
    z.bufi = 0

    # Remove old point
    x = z.x[!, z.i]
    z.ss -= sum(abs2, x)

    # Add new point
    x = z.x[!, z.i+z.n]
    z.ss += sum(abs2, x)
    z.σ = z isa SqNormNormalizer ? z.ss : sqrt(z.ss)
    z.i += 1
end


@inline @propagate_inbounds function getindex(z::AbstractNormNormalizer{T}, ::typeof(!), i::Int, inorder = i == z.bufi + 1) where T
    j = inorder ? i : z.n
    xj = z.i + i - 1
    σ = z.σ + eps(T)
    @avx for k = 1:size(z.x, 1)
        z.buffer[k, j] = z.x[k, xj] / σ
    end
    if inorder
        z.bufi = i
    end
    z.buffer[!, j]
end

LinearAlgebra.norm(z::NormNormalizer) = z.σ
LinearAlgebra.norm(z::SqNormNormalizer) = sqrt(z.σ)

Base.length(z::AbstractNormNormalizer{<:Any,2}) = size(z.x,1) * z.n
Base.length(z::AbstractNormNormalizer{<:Any,1}) = z.n
SlidingDistancesBase.lastlength(z::AbstractNormNormalizer) = z.n
actuallastlength(x::AbstractNormNormalizer) = lastlength(x.x)



# ============================================================================================

for T in [ZNormalizer, DiagonalZNormalizer, NormNormalizer, SqNormNormalizer]
    @eval @inline @propagate_inbounds function normalize(::Type{$T}, z::$T)
        if z.bufi == z.n
            return z.buffer
        end
        for i = z.bufi+1:z.n
            z[!, i, true] # This populates the buffer
        end
        return z.buffer
    end
end
