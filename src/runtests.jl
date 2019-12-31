using Plots, LaTeXStrings, Printf

include("tester.jl")


function runtests(n, γ, kmax; f_calls_limit=2000)
    U, G, H, ncm = genprob(n, γ, f_calls_limit=f_calls_limit)

    J = Symmetric(ones(n,n))
    ncm(G, J, kmax=3)
    Xold = copy(ncm.Xold)
    y = copy(ncm.res.y)

    ncm.Xold .= Xold
    tol = 1e-1
    t1 = @elapsed success, k = ncm(G, H, method=:IAPG,
                                   tol=tol,
                                   useXold=true,
                                   f_calls_limit=f_calls_limit)
    fgcount = ncm.res.fgcountRef[]
    r1 = ncm.res.resvals[1:fgcount]
    rp = ncm.res.rpRef[]
    rd = ncm.res.rdRef[]
    fval = ncm.res.fvals[fgcount]
    @printf("%4d %6.2f %8s %6d %8d %10.2e %10.2e %10.2e %8.2f\n",
            n, γ, "IAPG", k, fgcount, rp, rd, fval, t1)

    ncm.Xold .= Xold
    tol = 1e-1
    t2 = @elapsed success, k = ncm(G, H, method=:IR, τ=0.95,
                                   tol=tol,
                                   useXold=true,
                                   f_calls_limit=f_calls_limit)
    fgcount = ncm.res.fgcountRef[]
    r2 = ncm.res.resvals[1:fgcount]
    rp = ncm.res.rpRef[]
    rd = ncm.res.rdRef[]
    fval = ncm.res.fvals[fgcount]
    @printf("%4d %6.2f %8s %6d %8d %10.2e %10.2e %10.2e %8.2f\n",
            n, γ, "IR", k, fgcount, rp, rd, fval, t2)

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


function test(n, γ, kmax; f_calls_limit=2000)
    r1, r2 = runtests(n, γ, kmax, f_calls_limit=f_calls_limit)
    plt = makeplot(r1, r2)
    savefig(plt, "../figs/n$n-γ$γ-kmax$kmax.pdf")
    return nothing
end

################################################################################


@printf("%4s %6s %8s %6s %8s %10s %10s %10s %8s\n",
        "n", "γ", "method", "k", "fgcalls", "rp", "rd", "fval", "time")

kmax = 2000
for n = [587, 692, 834, 1255, 1869]
    for γ = [0.05, 0.1]
        test(n, γ, kmax, f_calls_limit=2000)
    end
end

