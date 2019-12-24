include("randncm.jl")
include("ProjPSD.jl")
include("calllbfgsb.jl")
include("ncm.jl")

function tester(n, γ; memlim=10, verbose=false)
    U, H = randncm(n, γ=γ)
    myproj = ProjPSD(n)
    storage = NCMstorage(n, memlim)

    @time ncm(U, H, myproj, storage, verbose=verbose)
    @time ncm(U, H, myproj, storage, method=:IR, τ=0.95, verbose=verbose)
    return nothing
end

tester(10, 0.1)
