include("helperfunctions.jl")
include("randncm.jl")
include("ProjPSD.jl")
include("calllbfgsb.jl")
include("ncm.jl")
include("dualobj.jl")

function genprob(
    n,
    γ;
    seed = 0,
    gaussian_noise = false,
    memlim = 10,
    maxfgcalls = 100_000,
)
    U, G, H = randncm(
        n,
        γ = γ,
        seed = seed,
        gaussian_noise = gaussian_noise,
    )

    ncm = NCM(
        n,
        memlim = memlim,
        maxfgcalls = maxfgcalls,
    )

    return U, G, H, ncm
end

function runall(
    G,
    H,
    ncm;
    maxfgcalls = 100_000,
    tol = 1e-1,
    printlevel = 1,
    useXold = true,
)
    if useXold
        X, y = CorNewton3(G)
        ncm.Xold .= X
    end

    @time ncm(
        G,
        H,
        method = :IAPG,
        maxfgcalls = maxfgcalls,
        tol = tol,
        printlevel = printlevel,
        useXold = useXold,
    )

    if useXold
        ncm.Xold .= X
    end

    @time ncm(
        G,
        H,
        method = :IR,
        τ = 0.95,
        maxfgcalls = maxfgcalls,
        tol = tol,
        printlevel = printlevel,
        useXold = useXold,
    )

    if useXold
        ncm.Xold .= X
    end

    H2 = ncm.H2
    H2.data .= H .^ 2
    L = fronorm(H2, ncm.proj.work)

    @time ncm(
        G,
        H,
        method = :IER,
        α = 19/L,
        σ = 1.0,
        maxfgcalls = maxfgcalls,
        tol = tol,
        printlevel = printlevel,
        useXold = useXold,
    )

    return nothing
end

function tester(
    n,
    γ;
    maxfgcalls = 100_000,
    seed = 0,
    gaussian_noise = false,
    tol = 1e-1,
    printlevel = 1,
    useXold = true,
)
    U, G, H, ncm = genprob(
        n,
        γ,
        seed = seed,
        gaussian_noise = gaussian_noise,
        maxfgcalls = maxfgcalls,
    )
    runall(
        G,
        H,
        ncm,
        maxfgcalls = maxfgcalls,
        tol = tol,
        printlevel = printlevel,
        useXold = useXold,
    )
end
