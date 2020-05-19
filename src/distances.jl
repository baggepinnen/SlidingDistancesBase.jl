"Z-normalized Euclidean distance"
struct ZEuclidean <: Distances.Metric end

function Distances.evaluate(d::ZEuclidean, x::AbstractArray{T}, y::AbstractArray{T}) where T
    mx,sx = meanstd(x)
    my,sy = meanstd(y)
    s = zero(float(T))
    @avx for i in eachindex(x,y)
        s += ((x[i]-mx)/sx - (y[i]-my)/sy)^2
    end
    √(s)
end

(d::ZEuclidean)(x, y) = evaluate(d, x, y)
