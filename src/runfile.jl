include("randncm.jl")
include("ProjPSD.jl")
include("calllbfgsb.jl")
include("ncm.jl")

function tester(n, γ;
        memlim=10,
        f_calls_limit=2000,
        tol=1e-2,
        verbose=false)

    U, H = randncm(n, γ=γ)
    myproj = ProjPSD(n)
    storage = NCMstorage(n,
        memlim=memlim, f_calls_limit=f_calls_limit)

    @time ncm(U, H, myproj, storage,
        f_calls_limit=f_calls_limit, tol=tol, verbose=verbose)
    @time ncm(U, H, myproj, storage, method=:IR, τ=0.95,
        f_calls_limit=f_calls_limit, tol=tol, verbose=verbose)

    return nothing
end

tester(10, 0.1)
