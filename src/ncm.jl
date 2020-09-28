using Printf

struct NCMresults
    X::Symmetric{Float64,Array{Float64,2}}
    y::Vector{Float64}
    Λ::Symmetric{Float64,Array{Float64,2}}
    fgcountRef::Base.RefValue{Int32}
    fvals::Vector{Float64}
    resvals::Vector{Float64}
    distvals::Vector{Float64}
    rpRef::Base.RefValue{Float64}
    rdRef::Base.RefValue{Float64}

    function NCMresults(n, maxfgcalls)
        X = Symmetric(zeros(n, n))
        y = zeros(n)
        Λ = Symmetric(zeros(n, n))
        fgcountRef = Ref{Int32}(0)
        fvals = Vector{Float64}(undef, maxfgcalls)
        resvals = Vector{Float64}(undef, maxfgcalls)
        distvals = Vector{Float64}(undef, maxfgcalls)
        rpRef = Ref{Float64}(0.0)
        rdRef = Ref{Float64}(0.0)

        new(X, y, Λ, fgcountRef, fvals, resvals, distvals, rpRef, rdRef)
    end
end


struct NCM
    n::Int32
    memlim::Int32
    maxfgcalls::Int32
    g::Vector{Float64}
    d::Vector{Float64}
    M::Symmetric{Float64,Array{Float64,2}}
    R::Symmetric{Float64,Array{Float64,2}}
    H2::Symmetric{Float64,Array{Float64,2}}
    Y::Symmetric{Float64,Array{Float64,2}}
    ∇fY::Symmetric{Float64,Array{Float64,2}}
    Γ::Symmetric{Float64,Array{Float64,2}}
    V::Symmetric{Float64,Array{Float64,2}}
    X::Symmetric{Float64,Array{Float64,2}}
    Z::Symmetric{Float64,Array{Float64,2}}
    Rd::Symmetric{Float64,Array{Float64,2}}
    Xold::Symmetric{Float64,Array{Float64,2}}
    wa::Vector{Float64}
    iwa::Vector{Int32}
    nbd::Vector{Int32}
    lower::Vector{Float64}
    upper::Vector{Float64}
    task::Vector{UInt8}
    task2::SubArray{UInt8,1,Vector{UInt8},Tuple{UnitRange{Int64}},true}
    csave::Vector{UInt8}
    lsave::Vector{Int32}
    isave::Vector{Int32}
    dsave::Vector{Float64}
    nRef::Base.RefValue{Int32}
    mRef::Base.RefValue{Int32}
    iprint::Base.RefValue{Int32}
    fRef::Base.RefValue{Float64}
    factr::Base.RefValue{Float64}
    pgtol::Base.RefValue{Float64}
    εRef::Base.RefValue{Float64}
    proj::ProjPSD
    res::NCMresults

    function NCM(n; memlim = 10, maxfgcalls = 100_000)
        g = zeros(n)
        d = zeros(n)

        M = Symmetric(zeros(n, n))
        R = copy(M)
        H2 = copy(M)
        Y = copy(M)
        ∇fY = copy(M)
        Γ = copy(M)
        V = copy(M)
        X = copy(M)
        Z = copy(M)
        Rd = copy(M)
        Xold = copy(M)

        nmax = n
        mmax = memlim

        wa = zeros(Cdouble, 2mmax * nmax + 5nmax + 11mmax * mmax + 8mmax)
        iwa = zeros(Cint, 3nmax)

        # provide nbd which defines the bounds on the variables:
        nbd = zeros(Cint, nmax)       # no bounds on the variables
        lower = zeros(Cdouble, nmax)    # the lower bounds
        upper = zeros(Cdouble, nmax)    # the upper bounds

        task = fill(Cuchar(' '), 60)   # fortran's blank padding
        task2 = view(task, 1:2)
        csave = fill(Cuchar(' '), 60)   # fortran's blank padding
        lsave = zeros(Cint, 4)
        isave = zeros(Cint, 44)
        dsave = zeros(Cdouble, 29)

        nRef = Ref{Cint}(n)
        mRef = Ref{Cint}(memlim)
        iprint = Ref{Cint}(-1)
        fRef = Ref{Cdouble}(0.0)
        factr = Ref{Cdouble}(0.0)
        pgtol = Ref{Cdouble}(0.0)

        εRef = Ref{Float64}(0.0)

        proj = ProjPSD(n)

        res = NCMresults(n, maxfgcalls)

        new(
            n,
            memlim,
            maxfgcalls,
            g,
            d,
            M,
            R,
            H2,
            Y,
            ∇fY,
            Γ,
            V,
            X,
            Z,
            Rd,
            Xold,
            wa,
            iwa,
            nbd,
            lower,
            upper,
            task,
            task2,
            csave,
            lsave,
            isave,
            dsave,
            nRef,
            mRef,
            iprint,
            fRef,
            factr,
            pgtol,
            εRef,
            proj,
            res,
        )
    end
end


function (ncm::NCM)(
    G::Symmetric{Float64,Array{Float64,2}},
    H::Symmetric{Float64,Array{Float64,2}};
    method::Symbol = :IAPG,
    τ::Float64 = 1.0,
    α::Float64 = 0.0,
    σ::Float64 = 1.0,
    tol::Float64 = 1e-1,
    kmax::Int64 = 100_000,
    maxfgcalls::Int64 = 100_000,
    printlevel::Int64 = 1,
    lbfgsbprintlevel::Int64 = -1,
    exact::Bool = false,
    useXold::Bool = false,
    innerverbose::Bool = false,
    cleanvals::Bool = true,
    scaleX::Bool = true,
)

    # Loss function and gradient
    #f(X) = 0.5*norm(H.*(X .- G))^2
    #∇f(X) = Symmetric(H2.*(X .- G))

    n = size(G, 1)
    validmethod = (method == :IAPG || method == :IR || method == :IER)

    # Check for valid input
    if maxfgcalls > ncm.maxfgcalls
        error("require maxfgcalls ≤ ncm.maxfgcalls")
    end
    validmethod || error("method must be :IAPG or :IR or :IER")
    n == ncm.n || error("require n == ncm.n")
    size(G) == size(H) || error("G and H must be the same size")
    issymmetric(G) || error("G must be symmetric")
    issymmetric(H) || error("H must be symmetric")
    !iszero(H) || error("H must be nonzero")

    g = ncm.g
    H2 = ncm.H2
    Y = ncm.Y
    V = ncm.V
    R = ncm.R
    Xold = ncm.Xold

    proj = ncm.proj
    res = ncm.res

    Xnew = res.X
    y = res.y
    Λ = res.Λ

    Xnew .= Xold

    fgcountRef = res.fgcountRef
    rpRef = res.rpRef
    rdRef = res.rdRef
    εRef = ncm.εRef
    fvals = res.fvals

    H2.data .= H .^ 2

    # Lipschitz constant of ∇f
    L = fronorm(H2, proj.work)

    if method == :IAPG
        τ == 1 || error("IAPG method requires τ = 1")
        t0 = 1.0
    end

    if method == :IR
        0 < τ ≤ 1 || error("IR method requires 0 < τ ≤ 1")
        0 ≤ α ≤ (1 - τ) * (L / τ) ||
        error("IR method requires 0 ≤ α ≤ $((1 - τ)*(L/τ))")
        t0 = 1.0
    end

    if method == :IER
        τ == 1 || error("IER method requires τ = 1")
        #α > 1 / L || error("IER method requires α > $(1/L)")
        α = 19/L
        α > 0 || error("IER method requires α > 0")
        0 ≤ σ ≤ 1 || error("IER method requires 0 ≤ σ ≤ 1")
        λ = α / (1 + α * L)
        t0 = 0.0
    end

    printlevel ≥ 1 && println("$method method, τ=$τ, α=$α, σ=$σ, tol=$tol")

    if !useXold
        fill!(Xold, 0.0)
    end

    fill!(y, 0.0)

    k = 0
    t = t0
    Y .= Xold
    innertol = NaN
    rp = rd = Inf
    innersuccess = true
    fgcountRef[] = 0
    fgcount = fgcountRef[]

    while (innersuccess &&
        max(rp, rd) > tol && k < kmax && fgcount < maxfgcalls
    )

        k += 1

        if method == :IER || method == :ER
            tnew = t + (λ + √(λ^2 + 4λ * t)) / 2
            Y.data .=
                (t / tnew) .* Xnew.data .+ ((tnew - t) / tnew) .* Xold.data
        else
            tnew = (1 + √(1 + 4 * t^2)) / 2
        end

        if method == :IAPG || method == :ER
            innertol = 1 / t^2
        end

        if exact
            innertol = 0.0
        end

        maxinnerfgcalls = maxfgcalls - fgcount

        # Solve the subproblem
        innersuccess = calllbfgsb!(
            ncm,
            G,
            H,
            tol,
            t,
            L,
            τ,
            α,
            σ;
            method = method,
            maxfgcalls = maxinnerfgcalls,
            innertol = innertol,
            exact = exact,
            verbose = innerverbose,
            lbfgsbprintlevel = lbfgsbprintlevel,
            cleanvals = cleanvals,
            scaleX = scaleX,
        )
        if !innersuccess
            printlevel ≥ 2 && println("Failed to solve subproblem.")
        end
        fgcount = fgcountRef[]
        innerfgcalls = fgcount - (maxfgcalls - maxinnerfgcalls)

        rp, rd, ε = rpRef[], rdRef[], εRef[]

        if printlevel ≥ 2
            mod(k, 20) == 1 && @printf(
                "%4s %8s %10s %10s %10s %10s %10s %10s\n",
                "k",
                "fgcalls",
                "||g||",
                "innertol",
                "f(X)",
                "rp",
                "rd",
                "ε"
            )
            @printf(
                "%4d %8d %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e\n",
                k,
                innerfgcalls,
                norm(g),
                innertol,
                fvals[fgcount],
                rp,
                rd,
                ε
            )
        end

        # Update
        if method == :IAPG
            Y.data .= Xnew.data .+ ((t - 1) / tnew) .* (Xnew.data .- Xold.data)
            Xold .= Xnew
        elseif method == :IR
            Y.data .=
                Xnew.data .- ((t / tnew) * (τ / L)) .* V.data .+
                ((t - 1) / tnew) .* (Xnew.data .- Xold.data)
            Xold .= Xnew
        elseif method == :IER
            Xold.data .-= (tnew - t) .* (V.data .+ (Y.data .- Xnew.data)./λ)
        end

        t = tnew
    end

    success = (max(rp, rd) ≤ tol)
    if printlevel ≥ 1
        resnorm = fronorm(R, proj.work)
        #pval = fvals[fgcount]
        #dval = 0.5*(dot(G, H2.*G) - dot(Xnew, H2.*Xnew)) + sum(y)
        println(success ? "Success." : "Failed.")
        @printf("%-20s: %-24d\n", "Outer iterations", k)
        @printf("%-20s: %-24d\n", "Function evals", fgcount)
        @printf("%-20s: %-24.16e\n", "Final ||H∘(X-G)||", resnorm)
        #@printf("%-20s: %-24.16e\n", "Primal objective",   pval)
        #@printf("%-20s: %-24.16e\n", "Dual objective",     dval)
        @printf("%-20s: %-24.6e\n", "Primal feasibility", rp)
        @printf("%-20s: %-24.6e\n", "Dual feasibility", rd)
        @printf("%-20s: %-24.6e\n", "Complementarity", symdot(Xnew, Λ))
        #@printf("%-20s: %-24d\n",    "rank(X)",            proj.m[])
    end

    return success, k
end
