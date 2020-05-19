# SlidingDistancesBase

[![Build Status](https://github.com/baggepinnen/SlidingDistancesBase.jl/workflows/CI/badge.svg)](https://github.com/baggepinnen/SlidingDistancesBase.jl/actions)
[![Coverage](https://codecov.io/gh/baggepinnen/SlidingDistancesBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/SlidingDistancesBase.jl)

This package defines some common functionality used to calculate a distance between windows sliding over vectors.

- `distance_profile(dist, query, timeseries)`
- `distance_profile!(D, dist, query, timeseries)`
- `ZEuclidean <: Distances.Metric` a Z-normalized Euclidean distance
- `sliding_mean!`
- `sliding_mean_std`


This package is used by
- [DynamicAxisWarping.jl](https://github.com/baggepinnen/DynamicAxisWarping.jl)
- [MatrixProfile.jl](https://github.com/baggepinnen/MatrixProfile.jl)
- [SpectralDistances.jl](https://github.com/baggepinnen/SpectralDistances.jl)
