include("randncm.jl")
include("ProjPSD.jl")
include("calllbfgsb.jl")
include("ncm.jl")

function genprob(n, γ, f_calls_limit; memlim=10)

    U, H = randncm(n, γ=γ)
    ncm = NCM(n,
              memlim=memlim,
              f_calls_limit=f_calls_limit)

    return U, H, ncm
end

function runall(U, H, ncm; tol=1e-2, verbose=false)

    H2 = ncm.H2
    H2.data .= H.^2
    L = fronorm(H2, ncm.proj.work)
    α = 2.0*(1/L)

    @time ncm(U, H, method=:IAPG,
        f_calls_limit=ncm.f_calls_limit,
        tol=tol, verbose=verbose)
    @time ncm(U, H, method=:IR, τ=0.95,
        f_calls_limit=ncm.f_calls_limit,
        tol=tol, verbose=verbose)
    @time ncm(U, H, method=:IER, α=α,
        f_calls_limit=ncm.f_calls_limit,
        tol=tol, verbose=verbose)

    return nothing
end

function tester(n, γ, f_calls_limit; tol=1e-2, verbose=false)
    U, H, ncm = genprob(n, γ, f_calls_limit)
    runall(U, H, ncm, tol=tol, verbose=verbose)
end

tester(10, 0.1, 100, verbose=true)

