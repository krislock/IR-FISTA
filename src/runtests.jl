using Plots, LaTeXStrings, Printf

include("tester.jl")


function runtests(n, γ; f_calls_limit=2000)
    U, H, ncm = genprob(n, γ, f_calls_limit=f_calls_limit)

    J = Symmetric(ones(n,n))
    ncm(U, J, kmax=3)
    Xold = copy(ncm.Xold)
    y = copy(ncm.res.y)

    tol = 1e-0
    t1 = @elapsed success = ncm(U, H, method=:IAPG,
                                useXold=true,
                                f_calls_limit=f_calls_limit)
    fgcount = ncm.res.fgcountRef[]
    r1 = ncm.res.resvals[1:fgcount]
    rp = ncm.res.rpRef[]
    rd = ncm.res.rdRef[]
    fval = ncm.res.fvals[fgcount]
    @printf("%4d %6.2f %8s %8d %10.2e %10.2e %16.8e %8.2f\n",
            n, γ, "IAPG", fgcount, rp, rd, fval, t1)

    tol = r1[end]
    ncm.Xold .= Xold
    ncm.res.y .= y
    t2 = @elapsed success = ncm(U, H, method=:IR, τ=0.95,
                                tol=tol,
                                useXold=true,
                                f_calls_limit=f_calls_limit)
    fgcount = ncm.res.fgcountRef[]
    r2 = ncm.res.resvals[1:fgcount]
    rp = ncm.res.rpRef[]
    rd = ncm.res.rdRef[]
    fval = ncm.res.fvals[fgcount]
    @printf("%4d %6.2f %8s %8d %10.2e %10.2e %16.8e %8.2f\n",
            n, γ, "IR", fgcount, rp, rd, fval, t2)

    return r1, r2
end


function makeplot(r1, r2)
    if r1[end] < 1e-2
        ylim = [1e-3, 1e1]
    elseif r1[end] < 1e-1
        ylim = [1e-2, 1e2]
    elseif r1[end] < 1e0
        ylim = [1e-1, 1e3]
    else
        ylim = [1e0, 1e4]
    end
    plt = plot(yaxis=:log,
               #xlim=[0, 2000],
               #ylim=ylim,
               xlabel="function evaluations",
               ylabel=L"\max\{r_p,r_d\}",
               size=(900,600))

    plot!(plt, r1, label="IAPG", ls=:dot,   lc=:black)
    plot!(plt, r2, label="IR",   ls=:solid, lc=:black)

    return plt
end


function test(n, γ; f_calls_limit=2000)
    r1, r2 = runtests(n, γ, f_calls_limit=f_calls_limit)
    plt = makeplot(r1, r2)
    savefig(plt, "../figs/n$n-γ$γ.pdf")
    return nothing
end

################################################################################


@printf("%4s %6s %8s %8s %10s %10s %16s %8s\n",
        "n", "γ", "method", "fgcalls", "rp", "rd", "fval", "time")

for n = [587, 692, 834, 1255, 1869]
    for γ = [0.01, 0.05, 0.1, 0.5]
        test(n, γ)
    end
end

