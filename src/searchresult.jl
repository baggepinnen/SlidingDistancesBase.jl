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


Base.@kwdef struct BatchSearchResult{T} <: AbstractSearchResult{T}
    target::T
    value
    ind::Int
    batch_ind::Int
    payload = nothing
end

Base.:(==)(a::BatchSearchResult,b::BatchSearchResult) = a.batch_ind == b.batch_ind && location(a) == location(b)
Base.getindex(v::Vector{String}, b::BatchSearchResult) = v[b.batch_ind]
