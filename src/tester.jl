include("helperfunctions.jl")
include("randncm.jl")
include("ProjPSD.jl")
include("calllbfgsb.jl")
include("ncm.jl")
include("dualobj.jl")

function genprob(n, γ;
                 seed=0,
                 memlim=10,
                 maxfgcalls=100_000)

    U, G, H = randncm(n, γ=γ, seed=seed)
    ncm = NCM(n,
              memlim=memlim,
              maxfgcalls=maxfgcalls)

    return U, G, H, ncm
end

function runall(G, H, ncm;
                maxfgcalls=100_000, tol=1e-2,
                printlevel=1)

    @time ncm(G, H, method=:IAPG,
              maxfgcalls=maxfgcalls, tol=tol,
              printlevel=printlevel)
    @time ncm(G, H, method=:IR, τ=0.95,
              maxfgcalls=maxfgcalls, tol=tol,
              printlevel=printlevel)
    #=
    H2 = ncm.H2
    H2.data .= H.^2
    L = fronorm(H2, ncm.proj.work)
    α = round(1/L, RoundUp, digits=2)
    @time ncm(G, H, method=:IER, α=α,
              maxfgcalls=maxfgcalls, tol=tol,
              printlevel=printlevel)
    =#

    return nothing
end

function tester(n, γ;
                maxfgcalls=100_000,
                seed=0,
                tol=1e-2,
                printlevel=1)

    U, G, H, ncm = genprob(n, γ,
                        seed=seed,
                        maxfgcalls=maxfgcalls)
    runall(G, H, ncm,
           maxfgcalls=maxfgcalls,
           tol=tol,
           printlevel=printlevel)
end

