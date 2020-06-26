module SlidingDistancesBase

using Statistics
using UnsafeArrays
using DocStringExtensions

export evaluate
using Distances

using LoopVectorization
using DSP


export ZEuclidean
include("distances.jl")

export lastlength, actuallastlength, getwindow, floattype
include("indexing.jl")

export distance_profile, distance_profile!, sliding_mean_std, sliding_mean!, sliding_entropy,
sliding_entropy_normalized
include("sliding.jl")

export AbstractNormalizer, ZNormalizer, DiagonalZNormalizer, normalize, znorm, meanstd, advance!
include("normalizers.jl")

export AbstractSearchResult, BatchSearchResult, value, location, payload, target, targetlength
include("searchresult.jl")


end
