using Plots, LaTeXStrings, Printf

include("runfile.jl")


function runtests(n, γ, kmax, f_calls_limit)
    U, H, ncm = genprob(n, γ, f_calls_limit)

    tol = 1e-2
    t1 = @elapsed success =
        ncm(U, H, method=:IAPG,
            kmax=kmax,
            tol=tol,
            f_calls_limit=f_calls_limit)
    fgcount = ncm.res.fgcountRef[]
    r1 = ncm.res.resvals[1:fgcount]
    @printf("%4d %6.2f %6d %8s %8d %10.2e %8.2f\n",
            n, γ, kmax, "IAPG", fgcount, r1[end], t1)

    tol = r1[end]
    t2 = @elapsed success =
        ncm(U, H, method=:IR, τ=0.95,
            tol=tol,
            f_calls_limit=f_calls_limit)
    fgcount = ncm.res.fgcountRef[]
    r2 = ncm.res.resvals[1:fgcount]
    @printf("%4d %6.2f %6d %8s %8d %10.2e %8.2f\n",
            n, γ, NaN, "IR", fgcount, r2[end], t2)

    return r1, r2
end


function makeplot(r1, r2)
    ylim = (r1[end] > 1e0) ? [1e0, 1e3] : [1e-1, 1e2]
    plt = plot(yaxis=:log,
               xlim=[0, 900],
               ylim=ylim,
               xlabel="function evaluations",
               ylabel=L"\max\{R_p,R_d\}",
               size=(900,600))

    plot!(plt, r1, label="IAPG", ls=:dot,   lc=:black)
    plot!(plt, r2, label="IR",   ls=:solid, lc=:black)

    return plt
end


function test(n, γ, kmax; f_calls_limit=2000)
    r1, r2 = runtests(n, γ, kmax, f_calls_limit)
    plt = makeplot(r1, r2)
    savefig(plt, "../figs/n$n-γ$γ-kmax$kmax.pdf")
    return nothing
end

############################################################


@printf("%4s %6s %6s %8s %8s %10s %8s\n",
        "n", "γ", "kmax", "method", "fgcalls", "resval", "time")

kmax = 300
#for n = [587, 692, 834, 1255, 1869]
for n = [100]
    for γ = [0.1, 0.05]
        test(n, γ, kmax)
    end
end

