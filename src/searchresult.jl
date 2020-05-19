abstract type AbstractSearchResult{T} end
value(r::AbstractSearchResult) = r.value
value(r::Number) = r
location(r::AbstractSearchResult) = r.ind
payload(r::AbstractSearchResult) = r.payload
target(r::AbstractSearchResult) = r.target
targetlength(r) = lastlength(target(r))
Base.isless(a::AbstractSearchResult,b::AbstractSearchResult) = isless(value(a), value(b))
Base.isless(a::Number,b::AbstractSearchResult) = isless(value(a), value(b))
Base.isless(a::AbstractSearchResult,b::Number) = isless(value(a), value(b))
Base.promote_rule(::Type{T}, ::Type{<:AbstractSearchResult}) where T <:Number = T
Base.convert(::Type{T}, r::AbstractSearchResult) where T <:Number = T(value(r))
Base.:*(n::Number, r::AbstractSearchResult) = n*value(r)


Base.getindex(v::T, s::AbstractSearchResult{T}) where {T<:AbstractArray} =
    getwindow(v, targetlength(s), location(s))


"""
    struct BatchSearchResult{T} <: AbstractSearchResult{T}

Stores the result after applying a search over a vector of vectors. An instance `bsr` of this type acts as a number in comparisons, and vectors of BatchSearchResult can be sorted based on the `value`. `BatchSearchResult` can also act like an index into vectors. If the indexed vector `v` is of same type as `target`, then a window into `v` is returned. If the vector is of some other eltype, then `v[bsr.batch_ind]` is returned.

See accessor functions [`value`](@ref), [`location`](@ref), [`payload`](@ref), [`target`](@ref), [`targetlength`](@ref)


# Arguments:
- `target`: Contains what was searched for
- `value`: Contains the value of what's found (like the distance or cost etc.)
- `ind`: Index into the searched vector at which the `value` was found.
- `batch_ind`: Index of outer vector (or file etc.)
- `payload`: extra information associated with the search.
"""
Base.@kwdef struct BatchSearchResult{T} <: AbstractSearchResult{T}
    target::T
    value
    ind::Int
    batch_ind::Int
    payload = nothing
end

Base.:(==)(a::BatchSearchResult,b::BatchSearchResult) = a.batch_ind == b.batch_ind && location(a) == location(b)
Base.getindex(v::Vector{String}, b::BatchSearchResult) = v[b.batch_ind]
