using SlidingDistancesBase
using Test

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

end
