using Plots, LaTeXStrings, Printf, Dates
using JLD

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
    printlevel = 0,
)
    methods = [:IR, :IAPG]
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
        #X, y = CorNewton3(G)
        X = load("CorNewton3solns/n$n-γ$γ.jld", "X")
    end

    H2 = ncm.H2
    H2.data .= H .^ 2
    L = fronorm(H2, ncm.proj.work)

    for method in methods
        plist = [NaN]
        if method == :IR
            L = norm(H.^2)
            τ, α = 0.9, 0.001*L
        elseif method == :IAPG
            τ, α = 1.0, 0.0
            plist = [2.0, 3.0, 4.0]
        end

        for p in plist
            if useXold
                ncm.Xold .= X
            end

            if method == :IR
                id = :IR
            elseif method == :IAPG
                if p == 2.0
                    id = :IAPG2
                elseif p == 3.0
                    id = :IAPG3
                elseif p == 4.0
                    id = :IAPG4
                else
                    error("p is not 2, 3, or 4")
                end
            end

            @printf("%4d %5.2f %7s ", n, γ, id)
            t = @elapsed success, k = ncm(
                G,
                H,
                method = method,
                p = p,
                τ = τ,
                α = α,
                tol = tol,
                useXold = useXold,
                maxfgcalls = maxfgcalls,
                printlevel = 0,
            )
            fgcount = ncm.res.fgcountRef[]
            results[id] = ncm.res.resvals[1:fgcount]
            rp = ncm.res.rpRef[]
            rd = ncm.res.rdRef[]
            @printf("%5d %6d %9.2e %9.2e %7.1f\n", k, fgcount, rp, rd, t)
        end
    end

    return results
end

function makeplot(results, tol)

    plt = plot(
        yaxis = :log,
        ylims = [1e-1, 1e+2],
        yticks = [1e-1, 1e+0, 1e+1, 1e+2],
        xlabel = "Total number of inner iterations",
        ylabel = L"\max\{r_p,r_d\}",
        size = (900, 600),
        ls = :auto,
        lc = :black,
    )

    plot!(plt, results[:IR],    label = "I-FISTA",        ls=:solid)
    plot!(plt, results[:IAPG2], label = "IA-FISTA (p=2)", ls=:dash)
    plot!(plt, results[:IAPG3], label = "IA-FISTA (p=3)", ls=:dot)
    plot!(plt, results[:IAPG4], label = "IA-FISTA (p=4)", ls=:dashdot)

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
    for n = 500:100:500
        for γ = 0.5:0.1:0.5
            test(n, γ)
        end
    end
end
println(time2str(t))

