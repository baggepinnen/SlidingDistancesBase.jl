using SlidingDistancesBase
using Distances
using Test, Statistics
using SlidingDistancesBase: window_dot

norm = x -> sqrt(sum(abs2, x))

@testset "SlidingDistancesBase.jl" begin

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


    @testset "ZEuclidean" begin
        @info "Testing ZEuclidean"
        d = ZEuclidean()
        x,y = randn(100) .+ 1, 2randn(100)
        @test evaluate(d,x,y) ≈ d(x,y) ≈ evaluate(Euclidean(), znorm(x), znorm(y))
    end


    @testset "generic distace_profile" begin
        @info "Testing generic distace_profile"

        dist = (x, y) -> norm(x - y)
        Q = randn(10)
        T = randn(11)
        P = distance_profile(dist, Q, T)
        @test P[1] == dist(Q, T[1:end-1])
        @test P[2] == dist(Q, T[2:end])
        P2 = distance_profile!(zeros(2), dist, Q, T)
        @test P2 ≈ P
        @test_throws ArgumentError distance_profile(dist, T, Q)
    end

    @testset "Euclidean distance_profile" begin
        @info "Testing Euclidean distance_profile"

       Q = randn(5)
       T = [randn(5); Q; randn(5)]
       D = distance_profile(ZEuclidean(), Q, T)
       @test D[6] < 1e-6
       @test D[1] ≈ norm(znorm(Q) - znorm(T[1:5]))
    end


    @testset "running stats" begin
        @info "Testing running stats"
        for l1 = [20,30]
            for l2 = [20,30]
                x = randn(l1)
                y = randn(l2)
                for w = 4:2:10
                    mx,sx = sliding_mean_std(x, w)
                    @test length(mx) == length(sx) == length(x)-w+1
                    @test mx[1] ≈ mean(x[1:w])
                    @test sx[1] ≈ std(x[1:w], corrected=false)

                    @test mx[2] ≈ mean(x[2:w+1])
                    @test sx[2] ≈ std(x[2:w+1], corrected=false)

                    @test sliding_mean!(similar(mx), x, w) ≈ mx

                    my,sy = sliding_mean_std(y, w)
                    QT = window_dot(getwindow(x,w,1), y)
                    D = distance_profile(ZEuclidean(), QT, mx,sx,my,sy,w)
                    @test D[1] ≈ ZEuclidean()(x[1:w], y[1:w])        atol=1e-5
                    @test D[2] ≈ ZEuclidean()(x[1:w], y[1 .+ (1:w)]) atol=1e-5
                    @test D[5] ≈ ZEuclidean()(x[1:w], y[4 .+ (1:w)]) atol=1e-5

                    QT = window_dot(getwindow(y,w,1), y)
                    D = distance_profile(ZEuclidean(), QT,my,sy,w)
                    @test D[1] == Inf # This is convention is this method to not return trivial matches
                    @test D[2] ≈ ZEuclidean()(y[1:w], y[1 .+ (1:w)]) atol=1e-5
                    @test D[5] ≈ ZEuclidean()(y[1:w], y[4 .+ (1:w)]) atol=1e-5

                end
            end
        end
    end


    @testset "SearchResultInterface" begin
        @info "Testing SearchResultInterface"

        tg        = randn(3)
        val       = 4
        ind       = 2
        batch_ind = 3
        pl        = randn(5)

        r = BatchSearchResult(tg, val, ind, batch_ind, pl)

        @test value(r)        == val
        @test value(1)        == 1
        @test location(r)     == ind
        @test payload(r)      == pl
        @test target(r)       == tg
        @test targetlength(r) == 3

        @test 1 < r < 100
        @test !(r < r)
        @test 2r == 8
        @test r == r
        @test sort([2r, r]) == [r, 2r]

        @test promote_rule(Float64, typeof(r)) == Float64
        @test convert(Float64, r) == val

        @test pl[r] == pl[ind:ind+length(tg)-1]
        @test ["1","2","3","4"][r] == "3"
    end

    @testset "Normalizers" begin
        @info "Testing Normalizers"
        include("test_normalizers.jl")

    end



end


# d = ZEuclidean()
# x = randn(100)
# @code_llvm meanstd(x)
# @code_llvm evaluate(d,x,x)
# @btime meanstd($x)
# @btime evaluate($d,$x,$x)
