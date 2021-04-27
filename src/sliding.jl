

"""
    distance_profile(dist, Q, T; kwargs...)

Calculate the distance profile corresponding to sliding `Q` over `T` and measuring distances between time windows using `dist`

The output is a vector of length `length(T) - length(Q) + 1`.
If inputs are matrices or higher-dimensional arrays, time is considered to be the last axis, and windows on the form `T[:, t:t+lastlength(Q)-1]`, where `lastlength` measures the length along the last axis, are compared.

Keyword arguments are sent to `dist`, which is called like this `dist(Q,T; kwargs...)`. `dist` can be a function or a distance from Distances.jl.

`distance_profile!(D, dist, Q, T; kwargs...)` is an in-place version.
"""
function distance_profile(dist, Q::AbstractArray{S}, T::AbstractArray{S}; kwargs...) where S
    m = lastlength(Q)
    n = lastlength(T)
    n >= m || throw(ArgumentError("Q cannot be longer than T"))
    l = n-m+1
    D = map(1:l) do i
        dist(Q, getwindow(T, m, i); kwargs...)
    end
end

"$(SIGNATURES)"
function distance_profile!(D::AbstractArray, dist, Q::AbstractArray{S}, T::AbstractArray{S}; kwargs...) where S
    m = lastlength(Q)
    n = lastlength(T)
    n >= m || throw(ArgumentError("Q cannot be longer than T"))
    l = n-m+1
    length(D) >= l || throw(ArgumentError("D is too short"))
    map!(D, 1:l) do i
        dist(Q, getwindow(T, m, i); kwargs...)
    end
end

"$(SIGNATURES)"
function distance_profile!(D::AbstractVector{S},::ZEuclidean, QT::AbstractVector{S}, μ, σ, m::Int, i::Int) where S <: Number
    @assert i <= length(D)
    @avx for j = eachindex(D)
        frac = (QT[j] - m*μ[i]*μ[j]) / (m*σ[i]*σ[j])
        D[j] = sqrt(max(2m*(1-frac), 0))
    end
    D[i] = typemax(eltype(D))
    D
end

"$(SIGNATURES)"
distance_profile(
    ::ZEuclidean,
    QT::AbstractVector{S},
    μ::AbstractVector{S},
    σ::AbstractVector{S},
    m::Int,
) where {S<:Number} = distance_profile!(similar(μ), ZEuclidean(), QT, μ, σ, m, 1)

"""
    $(SIGNATURES)

Accepts precomputed sliding mean and std of the input arrays. `QT` is the windowed dot product and `i` is the index into `T` (the longer time series).
"""
function distance_profile!(D::AbstractVector{S},::ZEuclidean, QT::AbstractVector{S}, μA, σA, μT, σT, m::Int, i::Int) where S <: Number
    @assert i <= length(μA)
    @avx for j = eachindex(D,QT,μT,σT)
        frac = (QT[j] - m*μA[i]*μT[j]) / (m*σA[i]*σT[j])
        D[j] = sqrt(max(2m*(1-frac), 0))
    end
    D
end

"$(SIGNATURES)"
distance_profile(::ZEuclidean, QT::AbstractVector{S}, μA, σA, μT, σT, m::Int) where {S<:Number} =
    distance_profile!(similar(μT), ZEuclidean(), QT, μA, σA, μT, σT, m, 1)


"""
    distance_profile(::ZEuclidean, Q, T)

Compute the z-normalized Euclidean distance profile corresponding to sliding `Q` over `T`
"""
function distance_profile!(
    D::AbstractVector{S},
    ::ZEuclidean,
    Q::AbstractVector{S},
    T::AbstractVector{S},
) where {S<:Number}
    m = length(Q)
    μ, σ = sliding_mean_std(T, m)
    QT = window_dot(znorm(Q), T) # TODO: allocates a new znorm(Q) each time
    @avx for j in eachindex(D,QT,σ)
        frac = QT[j] / (m * σ[j])
        D[j] = sqrt(max(2m * (1 - frac), 0))
    end
    D
end

"$(SIGNATURES)"
distance_profile(::ZEuclidean, Q::AbstractArray{S}, T::AbstractArray{S}) where {S} =
    distance_profile!(similar(T, length(T) - length(Q) + 1), ZEuclidean(), Q, T)

"""
The dot product between query Q and all subsequences of the same length as Q in time series T
"""
function window_dot(Q, T)
    n   = length(T)
    m   = length(Q)
    QT  = conv(reverse(Q), T)
    return QT[m:n]
end

"""
    $(SIGNATURES)

return vectors with mean and std of sliding windows of length `m`
"""
function sliding_mean_std(x::AbstractVector{T}, m) where T
    @assert length(x) >= m
    n = length(x)-m+1
    s = ss = zero(float(T))
    μ = Vector{float(T)}(undef, n)
    σ = Vector{float(T)}(undef, n)
    @avx for i = 1:m
        s  += x[i]
        ss += x[i]^2
    end
    μ[1] = s/m
    σ[1] = sqrt(max(ss/m - μ[1]^2, 0))
    @fastmath @inbounds for i = 1:n-1 # fastmath making it more accurate here as well, but not faster
        s -= x[i]
        ss -= x[i]^2
        s += x[i+m]
        ss += x[i+m]^2
        μ[i+1] = s/m
        σ[i+1] = sqrt(max(ss/m - μ[i+1]^2, 0))
    end
    μ,σ
end

@inbounds @fastmath function sliding_mean_std(x::AbstractMatrix{T}, m) where T
    @assert lastlength(x) >= m
    n = lastlength(x)-m+1
    s  = zeros(T, size(x,1))
    ss = zeros(T, size(x,1))
    μ = Matrix{float(T)}(undef, size(x,1), n)
    σ = Matrix{float(T)}(undef, size(x,1), n)
    for i = 1:m
        @avx s  .+= view(x, :, i)
        @avx ss .+= view(x, :, i).^2
    end
    @avx μ[:,1] .= s./m
    @avx σ[:,1] .= sqrt.(max.(ss./m .- view(μ, :, 1).^2, 0))
    for i = 1:n-1 # fastmath making it more accurate here as well, but not faster
        @avx s .-= view(x, :, i)
        @avx ss .-= view(x, :, i).^2
        @avx s .+= view(x, :, i+m)
        @avx ss .+= view(x, :, i+m).^2
        @avx μ[:,i+1] .= s./m
        @avx σ[:,i+1] .= sqrt.(max.(ss./m .- view(μ, :, i+1).^2, 0))
    end
    μ,σ
end

"""
    $(SIGNATURES)

return mean of sliding windows of length `m`. Operates in-place and stores result in first argument
"""
function sliding_mean!(μ,x::AbstractVector{T}, m) where T
    n = length(x)-m+1
    @assert length(x) >= m
    @assert length(μ) >= n

    s = zero(T)
    @avx for i = 1:m
        s += x[i]
    end
    μ[1] = s/m
    @fastmath @inbounds for i = 1:n-1 # fastmath making it more accurate here as well, but not faster
        s -= x[i]
        s += x[i+m]
        μ[i+1] = s/m
    end
    μ
end

"""
$(SIGNATURES)

Returns the sliding-window entropy
"""
function sliding_entropy(x::AbstractVecOrMat{T}, m=actuallastlength(x)) where T
    N = actuallastlength(x)
    n = N-m+1
    e = Vector{T}(undef, n)
    if x isa AbstractVector
        ent = vmap(x->x*log(x), x)
    else
        ent = zeros(T, N)
        @avx for i = 1:N
            for j = 1:size(x,1)
                ent[i] += x[j,i]*log(x[j,i])
            end
        end
    end
    e0 = zero(T)
    @avx for i = 1:m
        e0 += ent[i]
    end
    e[1] = e0
    @fastmath @inbounds for i = 1:n-1
        e0 -= ent[i]
        e0 += ent[i+m]
        e[i+1] = e0
    end
    e .= .-e
end

"""
$(SIGNATURES)

Returns the sliding-window entropy, normalized to have unit sum for each feature channel.
"""
function sliding_entropy_normalized(x::AbstractArray{T}, m=actuallastlength(x)) where T
    sums = sum(x, dims=ndims(x))
    sliding_entropy(x ./ sums, m)
end


"""
    $(SIGNATURES)

Sliding-window PCA. Returns `U1 ∈ 𝐑(N)` the main principal-component loading scaled with the singular value, and `V1 ∈ 𝐑(d × N)` the main principal direction.

# Arguments:
- `X`: Signal ∈ 𝐑(d × N)
- `fs`: Sampling frequency
- `w`: Window length in seconds
- `cutoff_freq`: Frequencies below this will be removed. Set to 0 to avoid filtering.
- `pad`: if true, the return value will be padded with zeros to account for the fact that the analysis uses sliding windows and produce a slightly shorter output. If `pad = true`, the output will have the same length as the input.
- `normalize_win`: remove the mean from each window before calculating svd.
- `center`: Index into `X` at the center of the window (default) or at a position in the window proportional to how far along in the signal the window is taken. 
"""
function sliding_pca(X::AbstractMatrix{A}; fs, w = 0.3, cutoff_freq = 0, pad = true, normalize_win=false, center = true) where A
    w = round(Int, w*fs) # from seconds to samples
    d,N = size(X)
    mX = mean(X, dims=2)
    X = X .- mX
    if cutoff_freq > 0
        X = filtfilt(digitalfilter(Highpass(cutoff_freq; fs), Butterworth(6)), X')'
    end

    win = 1:w
    dominant_mode = Vector{A}(undef, N-w+1)
    directions = Matrix{A}(undef, d, N-w+1)
    d1 = zeros(A, d); d1[1] = 1
    wininds = range(1, stop = w, length = N-w+1)
    for i = 1:N-w+1
        v = _maindir!(X[:, win], i == 1 ? d1 : @view(directions[:, i-1]), normalize_win)
        directions[:,i] .= v
        if center
            ii = (win[1] + win[end]) ÷ 2 + 1 # integer index from middle of window
            @views dominant_mode[i] = v'X[:, ii]
        else
            ii = i + wininds[i] - 1
            ib = floor(Int, ii)
            ia = ceil(Int, ii)
            α = ia - ii
            @views dominant_mode[i] = v'*((1-α) .* X[:, ib] .+ (α) .* X[:,ia]) # interpolate
        end
        win = win .+ 1
    end
    if cutoff_freq > 0
        directions = filtfilt(digitalfilter(Lowpass(cutoff_freq; fs), Butterworth(4)), directions')'
    end
    for i = axes(directions, 2)
        @views directions[:,i] ./= norm(directions[:,i])    
    end    
    if pad
        dominant_mode = [zeros(w÷2); dominant_mode; zeros(w÷2-1)]
        directions = [repeat(directions[:,1], 1, w÷2) directions repeat(directions[:,1], 1, w÷2-1)]
        directions
    end
    dominant_mode, directions
end

function _maindir!(x, last, normalize_win)
    if normalize_win
        x .-= mean(x, dims = 1)
    end
    s = svd!(x, alg = LinearAlgebra.QRIteration()) # alg choice important for stability of singular vectors
    V = s.U
    v = V[1,:] 
    if dot(last, v) < 0
        v .*= -1
    end
    v
end
