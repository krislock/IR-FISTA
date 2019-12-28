include("randncm.jl")
include("ProjPSD.jl")
include("calllbfgsb.jl")
include("ncm.jl")

function genprob(n, γ;
                 seed=0,
                 memlim=10,
                 f_calls_limit=2000)

    U, H = randncm(n, γ=γ, seed=seed)
    ncm = NCM(n,
              memlim=memlim,
              f_calls_limit=f_calls_limit)

    return U, H, ncm
end

function runall(U, H, ncm; tol=1e-2, printlevel=0)

    H2 = ncm.H2
    H2.data .= H.^2
    L = fronorm(H2, ncm.proj.work)
    α = round(1/L, RoundUp, digits=2)

    @time ncm(U, H, method=:IAPG,
        f_calls_limit=ncm.f_calls_limit, tol=tol, printlevel=printlevel)
    @time ncm(U, H, method=:IR, τ=0.95,
        f_calls_limit=ncm.f_calls_limit, tol=tol, printlevel=printlevel)
    @time ncm(U, H, method=:IER, α=α,
        f_calls_limit=ncm.f_calls_limit, tol=tol, printlevel=printlevel)

    return nothing
end

function tester(n, γ;
                f_calls_limit=2000,
                seed=0,
                tol=1e-2,
                printlevel=0)

    U, H, ncm = genprob(n, γ,
                        seed=seed,
                        f_calls_limit=f_calls_limit)
    runall(U, H, ncm,
           tol=tol,
           printlevel=printlevel)
end

