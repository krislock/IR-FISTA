using Plots, LaTeXStrings, Printf, Dates

ENV["GKSwstype"] = "nul"   # Removes plotting error when using VS Code remotely

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
    methods = [:IR, :IAPG]
    results = Dict{Symbol, Vector{Float64}}()

    U, G, H, ncm = genprob(
        n,
        γ,
        gaussian_noise = gaussian_noise,
        maxfgcalls = maxfgcalls,
    )

    if useXold
        X, y = CorNewton3(G)
    end

    for method in methods
        if method == :IR
            τ, α, σ = 0.95, 0.0, 1.0
        elseif method == :IAPG
            τ, α, σ = 1.0, 0.0, 1.0
        end

        if useXold
            ncm.Xold .= X
        end
        @printf("%4d %5.2f %7s ", n, γ, method)
        t = @elapsed success, k = ncm(
            G,
            H,
            method = method,
            τ = τ,
            α = α,
            σ = σ,
            tol = tol,
            useXold = useXold,
            maxfgcalls = maxfgcalls,
            printlevel = 0,
        )
        fgcount = ncm.res.fgcountRef[]
        results[method] = ncm.res.resvals[1:fgcount]
        rp = ncm.res.rpRef[]
        rd = ncm.res.rdRef[]
        @printf("%5d %6d %9.2e %9.2e %7.1f\n", k, fgcount, rp, rd, t)
    end

    return results
end

function makeplot(results)

    plt = plot(
        yaxis = :log,
        ylims = [1e-1, 1e+2],
        xlabel = "function evaluations",
        ylabel = L"\max\{r_p,r_d\}",
        size = (900, 600),
        ls = :auto,
        lc = :black,
    )

    plot!(plt, results[:IR],   label = "I-FISTA",  ls = :auto, lc = :black)
    plot!(plt, results[:IAPG], label = "IA-FISTA", ls = :auto, lc = :black)

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
    results = runtests(
        n,
        γ,
        gaussian_noise = gaussian_noise,
        tol = tol,
        maxfgcalls = maxfgcalls,
        useXold = useXold,
    )

    plt = makeplot(results)
    filename = @sprintf("n%d-γ%.2f.pdf", n, γ)
    savefig(plt, "../figs/$filename")

    return nothing
end

############################################################

@printf(
    "%4s %5s %7s %5s %6s %9s %9s %7s\n",
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
        for γ = 0.1:0.1:0.1
            test(n, γ)
        end
    end
end
println(time2str(t))
