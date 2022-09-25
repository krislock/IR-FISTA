
function myunstack(dfgrouped, method, category)
    df = dfgrouped[(method=method,)]
    ns = unique(df[!,:n])
    γs = unique(df[!,:γ])
    T = eltype(getproperty(df, category))
    A = Array{T}(unstack(df, :n, :γ, category))
    return NamedArray(A[:,2:end], (ns, γs), ("n", "γ"))
end

function make_performance_profile(df, category)

    IAPG2 = myunstack(df, "IAPG2", category)
    IAPG3 = myunstack(df, "IAPG3", category)
    IAPG4 = myunstack(df, "IAPG4", category)
    IR    = myunstack(df, "IR",    category)

    T = Float64[IR.array[:] IAPG2.array[:] IAPG3.array[:] IAPG4.array[:]]
    solvernames = ["I-FISTA", "IA-FISTA (p=2)", "IA-FISTA (p=3)", "IA-FISTA (p=4)"]

    plt = performance_profile(PlotsBackend(), T, solvernames, 
                              logscale=false, 
                              size=(800, 600),
                              legend=:bottomright,
                              linestyles=[:solid, :dash, :dot, :dashdot],
                              linecolor=:auto)

    pltfile = "performance_profile_$(category).pdf"
    savefig(plt, pltfile)
    println("Created $(pltfile)")

    return plt
end

if length(ARGS) != 1
    println("usage: julia make_performance_profiles.jl [resultsfile]")
else
    using CSV, DataFrames, NamedArrays
    using BenchmarkProfiles, Plots

    ENV["GKSwstype"] = "nul"   # Removes plotting error when using VS Code remotely

    resultsfile = ARGS[1]

    df = CSV.read(resultsfile, DataFrame, delim=" ", ignorerepeated=true)

    dfbymethod = groupby(df, :method)

    pptime = make_performance_profile(dfbymethod, :time)
    ppfgs  = make_performance_profile(dfbymethod, :fgs)
end

