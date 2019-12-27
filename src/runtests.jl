using Plots, LaTeXStrings

include("runfile.jl")

function runtests(n, γ, kmax, f_calls_limit)
    prob = genprob(n, γ, f_calls_limit)

    @time res = ncm(prob...,
                    method=:IAPG,
                    kmax=kmax,
                    f_calls_limit=f_calls_limit)
    r1 = res.resvals[1:res.fgcount[]]
    tol = r1[end]

    @time res = ncm(prob...,
                    method=:IR, τ=0.95,
                    tol=tol,
                    f_calls_limit=f_calls_limit)
    r2 = res.resvals[1:res.fgcount[]]

    return r1, r2
end


function makeplot(r1, r2)
    plt = plot(yaxis=:log,
               size=(900,600),
               xlabel="function evaluations",
               ylabel=L"\max\{R_p,R_d\}")

    plot!(plt, r1, label="AIPG", ls=:auto, lc=:black)
    plot!(plt, r2, label="IR",   ls=:auto, lc=:black)

    return plt
end

n, γ, kmax, f_calls_limit = 100, 0.1, 300, 1000
r1, r2 = runtests(n, γ, kmax, f_calls_limit)
plt = makeplot(r1, r2)
savefig(plt, "../figs/n$n-γ$γ-kmax$kmax.pdf")

#runtests(587, 0.1, 193, 1000)

