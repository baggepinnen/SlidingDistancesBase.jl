using SlidingDistancesBase
using Distances
using Test, Statistics

norm = x -> sqrt(sum(abs2, x))

@testset "SlidingDistancesBase.jl" begin
    @testset "distace_profile" begin
        @info "Testing distace_profile"

        dist = (x, y) -> norm(x - y)
        Q = randn(10)
        T = randn(11)
        P = distance_profile(dist, Q, T)
        @test P[1] == dist(Q, T[1:end-1])
        P2 = distance_profile!(zeros(2), dist, Q, T)
        @test P2 ≈ P
    end

    @testset "utils" begin
        @info "Testing utils"
        @test floattype(Int) == Float64
        @test floattype(Float32) == Float32
        @test floattype(missing) == Float64

        v = collect(1:3)
        @test v[!,1] == 1
        @test v[!,1:2] == 1:2

        v = copy([v v]')
        @test v[!,1] == [1,1]
        @test v[!,1:2] == [1 2; 1 2]

        v = cat(v,v,dims=3)
        @test v[!,1] == [1 2 3; 1 2 3]
        @test v[!,1:2] == cat([1 2 3; 1 2 3], [1 2 3; 1 2 3], dims=3)


        v = collect(1:3)
        v[!,1] = 5
        @test v[1] == 5
        v[!,1:2] = 5:6
        @test v[2] == 6

        v = copy([v v]')
        v[!,1] = [5,5]
        @test v[:,1] == [5,5]
        v[!,1:2] = [5 6; 5 6]
        @test v[:,1:2] == [5 6; 5 6]
        @test getwindow(v,2,1) == [5 6; 5 6]

        v = cat(v,v,dims=3)
        v[!,1] = [1 2 3; 1 2 3]
        v[!,1:2] = cat([1 2 3; 1 2 3], [1 2 3; 1 2 3], dims=3)

    end

    @testset "meanstd" begin
        @info "Testing meanstd"
        x = randn(100)
        m,s = meanstd(x)
        @test m ≈ mean(x)
        @test s ≈ std(x, corrected=false)
    end

    function znorm(x::AbstractVector)
        x = x .- mean(x)
        x ./= std(x, mean=0, corrected=false)
    end


    @testset "ZEuclidean" begin
        @info "Testing ZEuclidean"
        d = ZEuclidean()
        x,y = randn(100) .+ 1, 2randn(100)
        @test evaluate(d,x,y) ≈ d(x,y) ≈ evaluate(Euclidean(), znorm(x), znorm(y))
    end

end


# d = ZEuclidean()
# x = randn(100)
# @code_llvm meanstd(x)
# @code_llvm evaluate(d,x,x)
# @btime meanstd($x)
# @btime evaluate($d,$x,$x)
