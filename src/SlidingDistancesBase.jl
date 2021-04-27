module SlidingDistancesBase

using Statistics, LinearAlgebra
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
sliding_entropy_normalized,
sliding_pca
include("sliding.jl")

export AbstractNormalizer, ZNormalizer, DiagonalZNormalizer, NormNormalizer, SqNormNormalizer, normalize, znorm, meanstd, advance!
include("normalizers.jl")

export AbstractSearchResult, BatchSearchResult, value, location, payload, target, targetlength
include("searchresult.jl")


end
