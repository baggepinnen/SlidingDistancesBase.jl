using Distances
using Statistics
using SlidingDistancesBase
using SlidingDistancesBase: znorm

@test advance!(0) == 0
@test SlidingDistancesBase.setup_normalizer(Nothing, 1, 1) == (1, 1)
@test SlidingDistancesBase.normalize(Nothing, 1) == 1

@test SlidingDistancesBase.setup_normalizer(ZNormalizer, [1,0.5], [1,0.5])[1] == znorm([1,0.5])

n = 10
x = randn(100)
z = ZNormalizer(x,n)

@test length(z) == n
@test size(z) == (n,)
@test ndims(z) == 1
@test actuallastlength(z) == length(z.x)

advance!(z)  # Init
inds = 1:n
@test z == x[inds]
@test mean(z) ≈ mean(x[inds])
@test std(z) ≈ std(x[inds], corrected=false)
@test z.i == 1

@test @inferred(z[!,1]) == (x[1]-mean(z))/std(z)
@test @inferred(z[!,n]) == (x[n]-mean(z))/std(z)

advance!(z)
inds = inds .+ 1

@test z.i == 2
@test z == x[inds]
@test mean(z) ≈ mean(x[inds])
@test std(z) ≈ std(x[inds], corrected=false)

@test @inferred(z[!,1]) == (x[2]-mean(z))/std(z)
@test z[!,n] == (x[n+1]-mean(z))/std(z)



@test_throws BoundsError z[n+1]

@test normalize(ZNormalizer, z.x[2:11]) ≈ normalize(ZNormalizer, z)

for i = 1:89
    advance!(z)
end

@test z.bufi == 0
@test z[!, 1] ≈ (z[1]-mean(z))/std(z)
@test z.bufi == 1

@test normalize(ZNormalizer, z.x[91:end]) ≈ normalize(ZNormalizer, z) ≈ z.buffer
@test z.bufi == n

advance!(z)
@test_throws BoundsError z[n]



y = normalize(ZNormalizer, randn(10))
@test mean(y) ≈ 0 atol=1e-12
@test std(y, corrected=false) ≈ 1 atol=1e-12


## Diagonal ==================================================================================

n = 10
x = randn(2,100)
z = DiagonalZNormalizer(x,n)

@test length(z) == 2n
@test size(z) == (2,n)
@test ndims(z) == 2
@test actuallastlength(z) == size(z.x,2)

advance!(z)  # Init
inds = 1:n
@test Matrix(z) == x[:,inds]
@test mean(z) ≈ mean(x[:,inds], dims=2)
@test std(z) ≈ std(x[:,inds], corrected=false, dims=2)
@test z.i == 1
@test z[1,1] == x[1,1]
@test z[1:2,1] == x[1:2,1]


@test @inferred(z[!,1]) ≈ (x[:,1]-mean(z))./std(z)
@test @inferred(z[!,n]) ≈ (x[:,n]-mean(z))./std(z)

advance!(z)
inds = inds .+ 1

@test z[1,1] == x[1,2]
@test z[1:2,1] == x[1:2,2]

@test z.i == 2
@test Matrix(z) == x[:,inds]
@test mean(z) ≈ mean(x[:, inds], dims=2)
@test std(z) ≈ std(x[:, inds], dims=2, corrected=false)

@test @inferred(z[!,1]) ≈ (x[:,2]-mean(z))./std(z)
@test @inferred(z[!,n]) ≈ (x[:,n+1]-mean(z))./std(z)

@test z[1,1] == x[1,2]
@test z[1:2,1] == x[1:2,2]

@test normalize(DiagonalZNormalizer, z.x[:,2:11]) ≈ normalize(DiagonalZNormalizer, z)

for i = 1:89
    advance!(z)
end

@test z.bufi == 0
@test z[!, 1] ≈ (z[:,1]-mean(z))./std(z)
@test z.bufi == 1

@test normalize(DiagonalZNormalizer, z.x[:,91:end]) ≈ normalize(DiagonalZNormalizer, z) ≈ z.buffer
@test z.bufi == n

@test_throws BoundsError advance!(z)


y = normalize(DiagonalZNormalizer, randn(2,10))
@test mean(y) ≈ 0 atol=1e-12
@test std(y, corrected=false) ≈ 1 atol=1e-12


## Multidim ZNormalizer ==================================================================================

n = 10
x = 1 .+ randn(2,100)
z = ZNormalizer(x,n)

@test length(z) == 2n
@test size(z) == (2,n)
@test ndims(z) == 2
@test actuallastlength(z) == size(z.x,2)

advance!(z)  # Init
inds = 1:n
@test Matrix(z) == x[:,inds]
@test mean(z) ≈ mean(x[:,inds])
@test std(z) ≈ std(x[:,inds], corrected=false)
@test z.i == 1
@test z[1,1] == x[1,1]
@test z[1:2,1] == x[1:2,1]


@test @inferred(z[!,1]) ≈ (x[:,1].-mean(z))./std(z)
@test @inferred(z[!,n]) ≈ (x[:,n].-mean(z))./std(z)

advance!(z)
inds = inds .+ 1

@test z[1,1] == x[1,2]
@test z[1:2,1] == x[1:2,2]

@test z.i == 2
@test Matrix(z) == x[:,inds]
@test mean(z) ≈ mean(x[:, inds])
@test std(z) ≈ std(x[:, inds], corrected=false)

@test @inferred(z[!,1]) ≈ (x[:,2].-mean(z))./std(z)
@test @inferred(z[!,n]) ≈ (x[:,n+1].-mean(z))./std(z)

@test z[1,1] == x[1,2]
@test z[1:2,1] == x[1:2,2]

@test normalize(ZNormalizer, z.x[:,2:11]) ≈ normalize(ZNormalizer, z)

for i = 1:89
    advance!(z)
end

@test z.bufi == 0
@test z[!, 1] ≈ (z[:,1].-mean(z))./std(z)
@test z.bufi == 1

@test normalize(ZNormalizer, z.x[:,91:end]) ≈ normalize(ZNormalizer, z) ≈ z.buffer
@test z.bufi == n

y = normalize(ZNormalizer, randn(2,10))
@test mean(y) ≈ 0 atol=1e-12
@test std(y, corrected=false) ≈ 1 atol=1e-12


# NormNormalizer =============================================================================

@test SlidingDistancesBase.setup_normalizer(NormNormalizer, [1 0.5], [1 0.5])[1] ≈ [1 0.5]./norm([1 0.5])
@test SlidingDistancesBase.setup_normalizer(NormNormalizer, [1, 0.5], [1, 0.5])[1] ≈ [1, 0.5]./norm([1, 0.5])

n = 10
x = randn(2,100)
z = NormNormalizer(x,n)

@test length(z) == 2n
@test size(z) == (2,n)
@test ndims(z) == 2
@test actuallastlength(z) == size(z.x,2)

advance!(z)  # Init
inds = 1:n
@test Matrix(z) == x[:,inds]
@test z.i == 1
@test z[1,1] == x[1,1]
@test z[1:2,1] == x[1:2,1]


@test @inferred(z[!,1]) ≈ (x[:,1])./norm(z)
@test @inferred(z[!,n]) ≈ (x[:,n])./norm(z)

advance!(z)
inds = inds .+ 1

@test z[1,1] == x[1,2]
@test z[1:2,1] == x[1:2,2]

@test z.i == 2
@test Matrix(z) == x[:,inds]

@test @inferred(z[!,1]) ≈ (x[:,2])./norm(z)
@test @inferred(z[!,n]) ≈ (x[:,n+1])./norm(z)

@test z[1,1] == x[1,2]
@test z[1:2,1] == x[1:2,2]

@test normalize(NormNormalizer, z.x[:,2:11]) ≈ normalize(NormNormalizer, z)

for i = 1:89
    advance!(z)
end

@test z.bufi == 0
@test z[!, 1] ≈ (z[:,1])./norm(z)
@test z.bufi == 1

@test normalize(NormNormalizer, z.x[:,91:end]) ≈ normalize(NormNormalizer, z) ≈ z.buffer
@test z.bufi == n

@test_throws BoundsError advance!(z)


y = normalize(NormNormalizer, randn(2,10))
@test norm(y) ≈ 1 atol=1e-12


# SqNormNormalizer =============================================================================

@test SlidingDistancesBase.setup_normalizer(SqNormNormalizer, [1 0.5], [1 0.5])[1] ≈ [1 0.5]./norm([1 0.5])^2
@test SlidingDistancesBase.setup_normalizer(SqNormNormalizer, [1, 0.5], [1, 0.5])[1] ≈ [1, 0.5]./norm([1, 0.5])^2

n = 10
x = randn(2,100)
z = SqNormNormalizer(x,n)

@test length(z) == 2n
@test size(z) == (2,n)
@test ndims(z) == 2
@test actuallastlength(z) == size(z.x,2)

advance!(z)  # Init
inds = 1:n
@test Matrix(z) == x[:,inds]
@test z.i == 1
@test z[1,1] == x[1,1]
@test z[1:2,1] == x[1:2,1]


@test @inferred(z[!,1]) ≈ (x[:,1])./norm(z)^2
@test @inferred(z[!,n]) ≈ (x[:,n])./norm(z)^2

advance!(z)
inds = inds .+ 1

@test z[1,1] == x[1,2]
@test z[1:2,1] == x[1:2,2]

@test z.i == 2
@test Matrix(z) == x[:,inds]

@test @inferred(z[!,1]) ≈ (x[:,2])./norm(z)^2
@test @inferred(z[!,n]) ≈ (x[:,n+1])./norm(z)^2

@test z[1,1] == x[1,2]
@test z[1:2,1] == x[1:2,2]

@test normalize(SqNormNormalizer, z.x[:,2:11]) ≈ normalize(SqNormNormalizer, z)

for i = 1:89
    advance!(z)
end

@test z.bufi == 0
@test z[!, 1] ≈ (z[:,1])./norm(z)^2
@test z.bufi == 1

@test normalize(SqNormNormalizer, z.x[:,91:end]) ≈ normalize(SqNormNormalizer, z) ≈ z.buffer
@test z.bufi == n

@test_throws BoundsError advance!(z)


# y = normalize(SqNormNormalizer, randn(2,10))
# @test √norm(y) ≈ 1 atol=1e-12
