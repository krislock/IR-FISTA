include("randncm.jl")
include("ProjPSD.jl")
include("calllbfgsb.jl")
include("ncm.jl")

function genprob(n, γ, f_calls_limit; memlim=10)

    U, H = randncm(n, γ=γ)
    proj = ProjPSD(n)
    storage = NCMstorage(n,
        memlim=memlim,
        f_calls_limit=f_calls_limit)

    return U, H, proj, storage
end

function runall(U, H, proj, storage; tol=1e-2, verbose=false)

    H2 = storage.H2
    H2.data .= H.^2
    L = fronorm(H2, proj.work)
    α = nextfloat(1/L)

    @time ncm(U, H, proj, storage, method=:IAPG,
        f_calls_limit=storage.f_calls_limit,
        tol=tol, verbose=verbose)
    @time ncm(U, H, proj, storage, method=:IR, τ=0.95,
        f_calls_limit=storage.f_calls_limit,
        tol=tol, verbose=verbose)
    @time ncm(U, H, proj, storage, method=:IER, α=α,
        f_calls_limit=storage.f_calls_limit,
        tol=tol, verbose=verbose)

    return nothing
end

function tester(n, γ, f_calls_limit; tol=1e-2, verbose=false)
    U, H, proj, storage = genprob(n, γ, f_calls_limit)
    runall(U, H, proj, storage, tol=tol, verbose=verbose)
end

#tester(10, 0.1, 100)

