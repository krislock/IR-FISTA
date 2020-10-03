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
    printlevel = 0,
)
    methods = (:IR, :IER, :IAPG)
    results = Dict{Symbol, Vector{Float64}}()

    printlevel == 0 && @printf(
        "%4s %5s %7s %5s %6s %9s %9s %9s %7s\n",
        "n",
        "γ",
        "method",
        "k",
        "fgs",
        "rp",
        "rd",
        "ε",
        "time"
    )

    U, G, H, ncm = genprob(
        n,
        γ,
        gaussian_noise = gaussian_noise,
        maxfgcalls = maxfgcalls,
    )

    if useXold
        X, y = CorNewton3(G)
    end

    H2 = ncm.H2
    H2.data .= H .^ 2
    L = fronorm(H2, ncm.proj.work)

    for method in methods
        if method == :IR
            τ, α, σ = 0.95, 0.0, 1.0
        elseif method == :IER
            τ, α, σ = 1.0, 19/L, 1.0
        elseif method == :IAPG
            τ, α, σ = 1.0, 0.0, 1.0
        end

        if useXold
            ncm.Xold .= X
        end
        printlevel == 0 && @printf("%4d %5.2f %7s ", n, γ, method)
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
            printlevel = printlevel,
        )
        fgcount = ncm.res.fgcountRef[]
        results[method] = ncm.res.resvals[1:fgcount]
        rp = ncm.res.rpRef[]
        rd = ncm.res.rdRef[]
        ε  = ncm.res.εRef[]
        if printlevel == 0
            @printf("%5d %6d %9.2e %9.2e %9.2e %7.1f",
                    k, fgcount, rp, rd, ε, t)
            success || @printf(" <----- FAILED")
            @printf("\n")
        end
    end

    return results
end

function makeplot(results, tol)

    plt = plot(
        yaxis = :log,
        ylims = [tol, 1e+2],
        xlabel = "function evaluations",
        ylabel = L"\max\{r_p,r_d\}",
        size = (900, 600),
        ls = :auto,
        lc = :black,
    )

    method = (:IR, :IER, :IAPG)
    label  = ("I-FISTA", "IE-FISTA", "IA-FISTA")

    for i = 1:length(method)
        res = results[method[i]]
        plot!(plt, res, label=label[i], ls=:auto, lc=:black)

        # Put an "x" on the plot to indicate a method died
        fgs = length(res)
        finalres = res[end]
        if finalres > tol
            plot!(plt, [fgs], [finalres], m=:x, mc=:black, label=false)
        end
    end

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

    plt = makeplot(results, tol)
    filename = @sprintf("n%d-γ%.2f.pdf", n, γ)
    savefig(plt, "../figs/$filename")

    return nothing
end

############################################################

t = @elapsed begin
    for n = 100:100:800
        for γ = 0.1:0.1:1.0
            test(n, γ)
        end
    end
end
println(time2str(t))

