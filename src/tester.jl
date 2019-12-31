include("helperfunctions.jl")
include("randncm.jl")
include("ProjPSD.jl")
include("calllbfgsb.jl")
include("ncm.jl")
include("dualobj.jl")

function genprob(n, γ;
                 seed=0,
                 memlim=10,
                 f_calls_limit=2000)

    U, G, H = randncm(n, γ=γ, seed=seed)
    ncm = NCM(n,
              memlim=memlim,
              f_calls_limit=f_calls_limit)

    return U, G, H, ncm
end

function runall(G, H, ncm; tol=1e-2, printlevel=0)

    H2 = ncm.H2
    H2.data .= H.^2
    L = fronorm(H2, ncm.proj.work)
    α = round(1/L, RoundUp, digits=2)

    @time ncm(G, H, method=:IAPG,
              f_calls_limit=ncm.f_calls_limit, tol=tol, printlevel=printlevel)
    @time ncm(G, H, method=:IR, τ=0.95,
              f_calls_limit=ncm.f_calls_limit, tol=tol, printlevel=printlevel)
    #@time ncm(G, H, method=:IER, α=α,
    #          f_calls_limit=ncm.f_calls_limit, tol=tol, printlevel=printlevel)

    return nothing
end

function tester(n, γ;
                f_calls_limit=2000,
                seed=0,
                tol=1e-2,
                printlevel=0)

    U, G, H, ncm = genprob(n, γ,
                        seed=seed,
                        f_calls_limit=f_calls_limit)
    runall(G, H, ncm,
           tol=tol,
           printlevel=printlevel)
end

