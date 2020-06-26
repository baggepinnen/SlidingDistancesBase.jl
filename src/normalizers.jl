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

setup_normalizer(n::Val{Nothing}, q, y) = n, q, y

normalize(::Val{Nothing}, q) = q


normalize(T::Type, q) = normalize(Val(T), q) # TODO: use ::Type{T} instead and get rid of Val

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
mutable struct ZNormalizer{T} <: AbstractZNormalizer{T,1}
    x::Vector{T}
    n::Int
    μ::T
    σ::T
    s::T
    ss::T
    i::Int
    buffer::Vector{T}
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
    σ = sqrt(ss/n - μ^2)
    buffer = similar(x, n)
    ZNormalizer(x, n, μ, σ, s, ss, 0, buffer, 0)
end


function normalize(::Val{ZNormalizer}, q::AbstractVector)
    q = q .- mean(q)
    q ./= std(q, corrected=false, mean=0)
end

setup_normalizer(z::Val{ZNormalizer}, q::AbstractVector, y::AbstractVector) = z, normalize(z, q), ZNormalizer(y, length(q))

@propagate_inbounds function advance!(z::ZNormalizer{T}) where T

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
    z.σ = sqrt(z.ss/z.n - z.μ^2)
    z.i += 1
end


@inline @propagate_inbounds function getindex(z::ZNormalizer, i)
    @boundscheck 1 <= i <= z.n || throw(BoundsError(z,i))
    xi = i+z.i-1
    @boundscheck xi <= length(z.x) || throw(BoundsError(z,i))
    z.x[xi]
end

@inline @propagate_inbounds function getindex(z::ZNormalizer, ::typeof(!), i::Int, inorder = i == z.bufi + 1)
    y = (z[i]-z.μ) / z.σ
    if inorder
        z.bufi = i
        z.buffer[i] = y
    end
    y
end

@inline @propagate_inbounds function getindex(z::ZNormalizer, ::typeof(!), i::AbstractRange)
    @boundscheck (i[1] == z.i && length(i) == z.n) || throw(ArgumentError("ZNormalizers can only be indexed by ranges corresponding to their current state. Got range $i but state was $(z.i) corresponding to range $(z.i):$(z.i+z.n-1)"))
    z
end

Statistics.mean(z::AbstractZNormalizer) = z.μ
Statistics.std(z::AbstractZNormalizer) = z.σ

SlidingDistancesBase.lastlength(z::AbstractZNormalizer) = z.n
Base.length(z::AbstractZNormalizer) = z.n
Base.size(z::ZNormalizer) = (z.n,)
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
    σ = sqrt.(ss./n .- μ.^2)
    buffer = similar(x, size(x,1), n)
    DiagonalZNormalizer(x, n, μ, σ, s, ss, 0, buffer, 0)
end

function normalize(::Val{DiagonalZNormalizer}, q::AbstractMatrix)
    q = q .- mean(q, dims=2) # TODO: this will cause a ton of allocations
    q ./= (std(q, dims=2, corrected=false) .+ eps(eltype(q)))
end

setup_normalizer(z::Val{DiagonalZNormalizer}, q, y) = z, normalize(z, q), DiagonalZNormalizer(y, lastlength(q))
setup_normalizer(z::Val{ZNormalizer}, q::AbstractMatrix, y::AbstractMatrix) = setup_normalizer(Val(DiagonalZNormalizer), q, y) # Only need to expose ZNormalizer to the user


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



@inline @propagate_inbounds function getindex(z::DiagonalZNormalizer, i::Union{Number, AbstractRange}, j)
    @boundscheck 1 <= j <= z.n || throw(BoundsError(z,j))
    @boundscheck 1 <= 1 <= size(z.x, 1) || throw(BoundsError(z,i))
    xj = j+z.i-1
    @boundscheck xj <= lastlength(z.x) || throw(BoundsError(z,j))
    z.x[i, xj]
end

@inline @propagate_inbounds function getindex(z::DiagonalZNormalizer{T}, ::typeof(!), i, inorder = i == z.bufi + 1) where T
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

@inline @propagate_inbounds function getindex(z::DiagonalZNormalizer, ::typeof(!), i::AbstractRange)
    @boundscheck (i[1] == z.i && length(i) == z.n) || throw(ArgumentError("ZNormalizers can only be indexed by ranges corresponding to their current state. Got range $i but state was $(z.i) corresponding to range $(z.i):$(z.i+z.n-1)"))
    z
end

Base.Matrix(z::DiagonalZNormalizer) = z.x[:,z.i:z.i+z.n-1]

Base.length(z::DiagonalZNormalizer) = size(z.x,1) * z.n
Base.size(z::DiagonalZNormalizer) = (size(z.x,1), z.n)



for T in [ZNormalizer, DiagonalZNormalizer]
    @eval @inline @propagate_inbounds function normalize(::Val{$T}, z::$T)
        if z.bufi == z.n
            return z.buffer
        end
        for i = z.bufi+1:z.n
            z[!, i, true] # This populates the buffer
        end
        return z.buffer
    end
end
