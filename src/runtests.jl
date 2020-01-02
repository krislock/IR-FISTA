using Plots, LaTeXStrings, Printf, Dates

include("tester.jl")

function time2str(t)
    ms = Millisecond(1000round(t))
    dt = convert(DateTime, ms)
    tt = convert(Time, dt)
    return string(tt)
end

function runtests(n, γ; maxfgcalls=100_000)
    U, G, H, ncm = genprob(n, γ, maxfgcalls=maxfgcalls)

    J = Symmetric(ones(n,n))
    ncm(G, J, kmax=3, printlevel=0)
    Xold = copy(ncm.Xold)
    y = copy(ncm.res.y)

    ncm.Xold .= Xold
    tol = 1e-1
    t1 = @elapsed success, k = ncm(G, H, method=:IAPG,
                                   tol=tol,
                                   useXold=true,
                                   maxfgcalls=maxfgcalls,
                                   printlevel=0)
    fgcount = ncm.res.fgcountRef[]
    r1 = ncm.res.resvals[1:fgcount]
    rp = ncm.res.rpRef[]
    rd = ncm.res.rdRef[]
    fval = ncm.res.fvals[fgcount]
    @printf("%4d %6.2f %8s %6d %6d %10.2e %10.2e %10.2e %10s\n",
            n, γ, "IAPG", k, fgcount, rp, rd, fval, time2str(t1))

    ncm.Xold .= Xold
    tol = 1e-1
    t2 = @elapsed success, k = ncm(G, H, method=:IR, τ=0.95,
                                   tol=tol,
                                   useXold=true,
                                   maxfgcalls=maxfgcalls,
                                   printlevel=0)
    fgcount = ncm.res.fgcountRef[]
    r2 = ncm.res.resvals[1:fgcount]
    rp = ncm.res.rpRef[]
    rd = ncm.res.rdRef[]
    fval = ncm.res.fvals[fgcount]
    @printf("%4d %6.2f %8s %6d %6d %10.2e %10.2e %10.2e %10s\n",
            n, γ, "IR", k, fgcount, rp, rd, fval, time2str(t2))

    return r1, r2
end


function makeplot(r1, r2)

    plt = plot(yaxis=:log,
               ylims=[1e-1, 1e+1],
               xlabel="function evaluations",
               ylabel=L"\max\{r_p,r_d\}",
               size=(900,600))

    plot!(plt, r1, label="IAPG", ls=:dot,   lc=:black)
    plot!(plt, r2, label="IR",   ls=:solid, lc=:black)

    return plt
end


function test(n, γ; maxfgcalls=100_000)
    r1, r2 = runtests(n, γ, maxfgcalls=maxfgcalls)
    plt = makeplot(r1, r2)
    savefig(plt, "../figs/n$n-γ$γ.pdf")
    return nothing
end

################################################################################


@printf("%4s %6s %8s %6s %6s %10s %10s %10s %10s\n",
        "n", "γ", "method", "k", "fgs", "rp", "rd", "fval", "time")

#for n = [587, 692, 834, 1255, 1869]
for n = 500:500:2000
    for γ = [0.01, 0.05]
        test(n, γ)
    end
end

