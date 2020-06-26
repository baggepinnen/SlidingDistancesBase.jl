using Distances
using Statistics

@test advance!(0) == 0
@test SlidingDistancesBase.setup_normalizer(Val(Nothing), 1, 1) == (Val(Nothing), 1, 1)
@test SlidingDistancesBase.normalize(Val(Nothing), 1) == 1

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


## Multidim ==================================================================================

n = 10
x = randn(2,100)
z = IsoZNormalizer(x,n)

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

@test normalize(IsoZNormalizer, z.x[:,2:11]) ≈ normalize(IsoZNormalizer, z)

for i = 1:89
    advance!(z)
end

@test z.bufi == 0
@test z[!, 1] ≈ (z[:,1]-mean(z))./std(z)
@test z.bufi == 1

@test normalize(IsoZNormalizer, z.x[:,91:end]) ≈ normalize(IsoZNormalizer, z) ≈ z.buffer
@test z.bufi == n

@test_throws BoundsError advance!(z)




y = normalize(IsoZNormalizer, randn(2,10))
@test mean(y) ≈ 0 atol=1e-12
@test std(y, corrected=false) ≈ 1 atol=1e-12
