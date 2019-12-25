using LinearAlgebra, Random
using Statistics, Distributions

"""
@article{Lewandowski2009a,
    Author = {Daniel Lewandowski and Dorota Kurowicka and Harry Joe},
    Journal = {Journal of Multivariate Analysis},
    Number = {9},
    Pages = {1989 - 2001},
    Title = {Generating random correlation matrices based on vines and extended onion method},
    Volume = {100},
    Year = {2009}}
https://doi.org/10.1016/j.jmva.2009.04.008
"""
function onion(n; η = 1.0)
    S = Symmetric(ones(n, n))
    β = η + (n-2)/2
    S.data[1,2] = 2*rand(Beta(β,β)) - 1
    for k = 2:n-1
        β -= 0.5
        y = rand(Beta(k/2, β))
        u = normalize!(randn(k))
        w = sqrt(y)*u
        F = cholesky(Symmetric(S[1:k,1:k]))
        S.data[1:k,k+1] = F.L*w
    end
    return S
end

function randncm(n; method=:onion, seed=0, γ=0.0, p=0.5)
    rng = Random.seed!(seed)

    # Random target matrix
    if method == :onion
        U = onion(n)
    else
        U = 2*rand(n,n) .- 1; U = Symmetric(triu(U,1) + I)
    end

    # Random noise
    E = Symmetric(triu(randn(n,n),1))
    U.data .= (1-γ).*U .+ γ.*E
    U = Symmetric(triu(U,1) + I)

    # Random sparse H
    H = [rand()<p ? 1.0 : 0.0 for i=1:n, j=1:n]
    H = Symmetric(triu(H,1))

    Random.seed!(rng)

    return U, H
end
