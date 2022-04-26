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
    β = η + (n - 2) / 2
    S.data[1, 2] = 2 * rand(Beta(β, β)) - 1
    for k = 2:n-1
        β -= 0.5
        y = rand(Beta(k / 2, β))
        u = normalize!(Random.randn(k))
        w = sqrt(y) * u
        F = cholesky(Symmetric(S[1:k, 1:k]))
        S.data[1:k, k+1] = F.L * w
    end
    return S
end

function randncm(n; seed = 0, γ = 0.0, p = 0.5, gaussian_noise = false)
    rng = Random.seed!(seed)

    # Random target matrix
    U = onion(n)

    # Random sparse H
    if gaussian_noise
        # Hij ∈ [1, 10] for Hij ≠ 0
        Htmp = [rand() < p ? 1.0 + 9.0 * rand() : 0.0 for i = 1:n, j = 1:n]
    else
        Htmp = [rand() < p ? rand() : 0.0 for i = 1:n, j = 1:n]
    end
    H = Symmetric(triu(Htmp, 1) + I)

    # Random noise
    if gaussian_noise
        # G = U + E./H
        G = copy(U)
        for j = 1:n
            for i = 1:j-1
                Hij = H.data[i, j]
                if Hij > 0.0
                    G.data[i, j] += γ * Random.randn() / Hij
                else
                    G.data[i, j] = 0.0
                end
            end
        end
    else
        Etmp = 2 * rand(n, n) .- 1
        E = Symmetric(triu(Etmp, 1) + I)
        Gtmp = (1 - γ) .* U .+ γ .* E
        G = Symmetric(triu(Gtmp, 1) + I)
    end

    Random.seed!(rng)

    return U, G, H
end
