using Plots, LaTeXStrings, Printf, Dates

include("tester.jl")

function time2str(t)
    ms = Millisecond(1000round(t))
    dt = convert(DateTime, ms)
    tt = convert(Time, dt)
    return string(tt)
end

function runtests(n, γ;
                  gaussian_noise=false,
                  tol=1e-1,
                  maxfgcalls=100_000,
                  useXold=true)

    U, G, H, ncm = genprob(n, γ,
                           gaussian_noise=gaussian_noise,
                           maxfgcalls=maxfgcalls)

    if useXold
        X, y = CorNewton3(G)
        ncm.Xold .= X
    end

    @printf("%4d %6.2f %8s ", n, γ, "IAPG")
    t1 = @elapsed success, k = ncm(G, H, method=:IAPG,
                                   tol=tol,
                                   useXold=useXold,
                                   maxfgcalls=maxfgcalls,
                                   printlevel=0)
    fgcount = ncm.res.fgcountRef[]
    r1 = ncm.res.resvals[1:fgcount]
    rp = ncm.res.rpRef[]
    rd = ncm.res.rdRef[]
    @printf("%6d %6d %10.2e %10.2e %10s\n",
            k, fgcount, rp, rd, time2str(t1))

    if useXold
        ncm.Xold .= X
    end

    @printf("%4d %6.2f %8s ", n, γ, "IR")
    t2 = @elapsed success, k = ncm(G, H, method=:IR, τ=0.95,
                                   tol=tol,
                                   useXold=useXold,
                                   maxfgcalls=maxfgcalls,
                                   printlevel=0)
    fgcount = ncm.res.fgcountRef[]
    r2 = ncm.res.resvals[1:fgcount]
    rp = ncm.res.rpRef[]
    rd = ncm.res.rdRef[]
    @printf("%6d %6d %10.2e %10.2e %10s\n",
            k, fgcount, rp, rd, time2str(t2))

    return r1, r2
end

function makeplot(r1, r2)

    plt = plot(yaxis=:log,
               ylims=[1e-1, 1e+2],
               xlabel="function evaluations",
               ylabel=L"\max\{r_p,r_d\}",
               size=(900,600))

    plot!(plt, r1, label="IAPG", ls=:dot,   lc=:black)
    plot!(plt, r2, label="IR",   ls=:solid, lc=:black)

    return plt
end

function test(n, γ;
              gaussian_noise=false,
              tol=1e-1,
              maxfgcalls=100_000)
    r1, r2 = runtests(n, γ,
                      gaussian_noise=gaussian_noise,
                      tol=tol,
                      maxfgcalls=maxfgcalls)
    plt = makeplot(r1, r2)
    filename = @sprintf("n%d-γ%.2f.pdf", n, γ)
    savefig(plt, "../figs/$filename")
    return nothing
end

############################################################

@printf("%4s %6s %8s %6s %6s %10s %10s %10s\n",
        "n", "γ", "method", "k", "fgs", "rp", "rd", "time")
for n = 100:100:1000
    for γ = 0.1:0.1:1.0
        test(n, γ, tol=1e-1)
    end
end

