using Plots, LaTeXStrings, Printf, Dates

include("tester.jl")

function time2str(t)
    ms = Millisecond(round(1000 * t))
    dt = convert(DateTime, ms)
    tt = convert(Time, dt)
    return string(tt)
end

function runtests(
    n,
    γ;
    gaussian_noise = false,
    tol = 1e-1,
    maxfgcalls = 100_000,
    useXold = true,
)

    U, G, H, ncm = genprob(
        n,
        γ,
        gaussian_noise = gaussian_noise,
        maxfgcalls = maxfgcalls,
    )

    if useXold
        X, y = CorNewton3(G)
        ncm.Xold .= X
    end
    @printf("%4d %6.2f %8s ", n, γ, "IAPG")
    t = @elapsed success, k = ncm(
        G,
        H,
        method = :IAPG,
        tol = tol,
        useXold = useXold,
        maxfgcalls = maxfgcalls,
        printlevel = 0,
    )
    fgcount = ncm.res.fgcountRef[]
    IAPGresvals = ncm.res.resvals[1:fgcount]
    rp = ncm.res.rpRef[]
    rd = ncm.res.rdRef[]
    @printf("%6d %6d %10.2e %10.2e %12.1f\n", k, fgcount, rp, rd, t)

    if useXold
        ncm.Xold .= X
    end
    @printf("%4d %6.2f %8s ", n, γ, "IR")
    t = @elapsed success, k = ncm(
        G,
        H,
        method = :IR,
        τ = 0.95,
        tol = tol,
        useXold = useXold,
        maxfgcalls = maxfgcalls,
        printlevel = 0,
    )
    fgcount = ncm.res.fgcountRef[]
    IRresvals = ncm.res.resvals[1:fgcount]
    rp = ncm.res.rpRef[]
    rd = ncm.res.rdRef[]
    @printf("%6d %6d %10.2e %10.2e %12.1f\n", k, fgcount, rp, rd, t)

    if useXold
        ncm.Xold .= X
    end
    @printf("%4d %6.2f %8s ", n, γ, "IER")
    H2 = ncm.H2
    H2.data .= H .^ 2
    L = fronorm(H2, ncm.proj.work)
    α = round(1 / L, RoundUp, digits = 2)
    t = @elapsed success, k = ncm(
        G,
        H,
        method = :IER,
        α = α,
        σ = 1.0,
        tol = tol,
        useXold = useXold,
        maxfgcalls = maxfgcalls,
        printlevel = 0,
    )
    fgcount = ncm.res.fgcountRef[]
    IERresvals = ncm.res.resvals[1:fgcount]
    rp = ncm.res.rpRef[]
    rd = ncm.res.rdRef[]
    @printf("%6d %6d %10.2e %10.2e %12.1f\n", k, fgcount, rp, rd, t)

    return IAPGresvals, IRresvals, IERresvals
end

function makeplot(IAPGresvals, IRresvals, IERresvals)

    plt = plot(
        yaxis = :log,
        ylims = [1e-1, 1e+2],
        xlabel = "function evaluations",
        ylabel = L"\max\{r_p,r_d\}",
        size = (900, 600),
        ls = :auto,
        lc = :black,
    )

    plot!(plt, IAPGresvals, label = "IAPG", ls = :auto, lc = :black)
    plot!(plt, IRresvals, label = "IR", ls = :auto, lc = :black)
    plot!(plt, IERresvals, label = "IER", ls = :auto, lc = :black)

    return plt
end

function test(
    n,
    γ;
    gaussian_noise = false,
    tol = 1e-1,
    maxfgcalls = 100_000,
    useXold = true,
)
    resvals = runtests(
        n,
        γ,
        gaussian_noise = gaussian_noise,
        tol = tol,
        maxfgcalls = maxfgcalls,
        useXold = useXold,
    )
    plt = makeplot(resvals...)
    filename = @sprintf("n%d-γ%.2f.pdf", n, γ)
    savefig(plt, "../figs/$filename")
    return nothing
end

############################################################

@printf(
    "%4s %6s %8s %6s %6s %10s %10s %12s\n",
    "n",
    "γ",
    "method",
    "k",
    "fgs",
    "rp",
    "rd",
    "time"
)
t = @elapsed begin
    for n = 100:100:100
        for γ = 0.1:0.1:1.0
            test(n, γ)
        end
    end
end
println(time2str(t))
