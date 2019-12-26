const NCMmethods = [:IAPG, :IR, :IER]


struct NCMresults
    X::Symmetric{Float64,Array{Float64,2}}
    y::Vector{Float64}
    Λ::Symmetric{Float64,Array{Float64,2}}
    fgcount::Base.RefValue{Int32}
    fvals::Vector{Float64}
    resvals::Vector{Float64}

    function NCMresults(n, f_calls_limit)
        X = Symmetric(zeros(n,n))
        y = zeros(n)
        Λ = Symmetric(zeros(n,n))
        fgcount = Ref{Int32}(0)
        fvals   = Vector{Float64}(undef, f_calls_limit)
        resvals = Vector{Float64}(undef, f_calls_limit)

        new(X, y, Λ, fgcount, fvals, resvals)
    end
end


struct NCMstorage
    n::Int32
    memlim::Int32
    f_calls_limit::Int32
    g::Vector{Float64}
    d::Vector{Float64}
    M::Symmetric{Float64,Array{Float64,2}}
    H2::Symmetric{Float64,Array{Float64,2}}
    Y::Symmetric{Float64,Array{Float64,2}}
    ∇fY::Symmetric{Float64,Array{Float64,2}}
    Γ::Symmetric{Float64,Array{Float64,2}}
    V::Symmetric{Float64,Array{Float64,2}}
    X::Symmetric{Float64,Array{Float64,2}}
    Xold::Symmetric{Float64,Array{Float64,2}}
    Z::Symmetric{Float64,Array{Float64,2}}
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
    rpRef::Base.RefValue{Float64}
    rdRef::Base.RefValue{Float64}
    εRef::Base.RefValue{Float64}
    δRef::Base.RefValue{Float64}
    βRef::Base.RefValue{Float64}
    distRef::Base.RefValue{Float64}
    res::NCMresults

    function NCMstorage(n; memlim=10, f_calls_limit=2000)
        g = zeros(n)
        d = zeros(n)

        M = Symmetric(zeros(n,n))
        H2 = copy(M)
        Y = copy(M)
        ∇fY = copy(M)
        Γ = copy(M)
        V = copy(M)
        X = copy(M)
        Xold = copy(M)
        Z = copy(M)

        nmax = n
        mmax = memlim

        wa = zeros(Cdouble, 2mmax*nmax + 5nmax + 11mmax*mmax + 8mmax)
        iwa = zeros(Cint, 3nmax)

        # provide nbd which defines the bounds on the variables:
        nbd   = zeros(Cint, nmax)       # no bounds on the variables
        lower = zeros(Cdouble, nmax)    # the lower bounds
        upper = zeros(Cdouble, nmax)    # the upper bounds

        task  = fill(Cuchar(' '), 60)   # fortran's blank padding
        task2 = view(task, 1:2)
        csave = fill(Cuchar(' '), 60)   # fortran's blank padding
        lsave = zeros(Cint, 4)
        isave = zeros(Cint, 44)
        dsave = zeros(Cdouble, 29)

        nRef   = Ref{Cint}(n)
        mRef   = Ref{Cint}(memlim)
        iprint = Ref{Cint}(-1)
        fRef  = Ref{Cdouble}(0.0)
        factr = Ref{Cdouble}(0.0)
        pgtol = Ref{Cdouble}(0.0)

        rpRef   = Ref{Float64}(0.0)
        rdRef   = Ref{Float64}(0.0)
        εRef    = Ref{Float64}(0.0)
        δRef    = Ref{Float64}(0.0)
        βRef    = Ref{Float64}(0.0)
        distRef = Ref{Float64}(0.0)

        res = NCMresults(n, f_calls_limit)

        new(n, memlim, f_calls_limit,
            g, d, M, H2, Y, ∇fY, Γ, V, X, Xold, Z,
            wa, iwa, nbd, lower, upper, task, task2, csave, lsave, isave, dsave,
            nRef, mRef, iprint, fRef, factr, pgtol,
            rpRef, rdRef, εRef, δRef, βRef, distRef, res)
    end
end


function fronorm(A, work)
    lda, n = size(A)
    ccall((:dlansy_64_, "libopenblas64_"), Cdouble,
        (Ref{UInt8}, Ref{UInt8}, Ref{Int}, Ptr{Float64}, Ref{Int},
        Ptr{Float64}), 'F', A.uplo, n, A.data, lda, work)
end


function symmetrize!(A)
   n = size(A,1)
   for j=1:n
       for i=1:j-1
           A.data[j,i] = A.data[i,j]
       end
   end
   return nothing
end


function symdot(A, B)
    symmetrize!(A)
    symmetrize!(B)
    return dot(A.data, B.data)
end


# Evaluates dual objective function and its gradient
function dualobj!(gg, y, proj,
        n, H, H2, Y, U, ∇fY, M, X, Λ, Γ, d, Xnew, V, Z,
        fgcount, fvals, resvals,
        rpRef, rdRef, εRef, δRef, distRef, L, τ)

    fgcount[] += 1

    τdL = τ/L
    Ldτ = L/τ

    # ∇fY.data .= H2.*(Y .- U)
    # M .= ∇fY; plusdiag!(M, y)  # M = ∇f(Y) + Diag(y)
    # M.data .= Y .- τdL.*M      # M = Y - (τ/L)*(∇f(Y) + Diag(y))
    # X .= M
    @inbounds for j=1:n
        for i=1:j
            ∇fY.data[i,j] = H2.data[i,j]*(Y.data[i,j] - U.data[i,j])
            M.data[i,j] = Y.data[i,j] - τdL*∇fY.data[i,j]
            X.data[i,j] = M.data[i,j]
        end
        M.data[j,j] -= τdL*y[j]
        X.data[j,j] = M.data[j,j]
    end

    proj(X)

    # Λ.data .= Ldτ.*(X .- M)         # Λ is psd
    # Γ.data .= .-Λ; plusdiag!(Γ, y)  # Γ = Diag(y) - Λ

    # Ensure that diag(Xnew).==1 exactly
    @inbounds for j=1:n
        if X.data[j,j] > 0.0
            d[j] = 1.0/sqrt(X.data[j,j])
        else
            d[j] = 1.0
        end
    end
    @inbounds for j=1:n
        for i=1:j
            Λ.data[i,j] = Ldτ*(X.data[i,j] - M.data[i,j])
            Γ.data[i,j] = -Λ.data[i,j]
            Xnew.data[i,j] = d[i]*d[j]*X.data[i,j]
            Z.data[i,j] = Xnew.data[i,j] - Y.data[i,j]
            V.data[i,j] = ∇fY.data[i,j] + Ldτ*Z.data[i,j] + Γ.data[i,j]
            M.data[i,j] = H.data[i,j]*(Xnew.data[i,j] - U.data[i,j])
        end
        Γ.data[j,j] += y[j]
        V.data[j,j] += y[j]
    end

    # V.data .= ∇fY .+ Ldτ.*(Xnew .- Y) .+ Γ

    # Compute and store the objective function
    # M.data .= H.*(Xnew .- U)
    fvals[fgcount[]] = 0.5*fronorm(M, proj.work)^2

    # Compute and store the optim. cond. residual
    @inbounds for j=1:n
        d[j] = 1.0 - Xnew[j,j]
        for i=1:j
            M.data[i,j] = H.data[i,j]*M.data[i,j] + Γ.data[i,j]
        end
    end
    rpRef[] = norm(d)/(1 + √n)
    #M.data .= H2.*(Xnew .- U) .+ Γ
    rdRef[] = fronorm(M, proj.work)
    resvals[fgcount[]] = max(rpRef[],rdRef[])

    εRef[]    = symdot(Xnew, Λ)
    δRef[]    = fronorm(V, proj.work)
    distRef[] = fronorm(Z, proj.work)

    # Compute the gradient of the dual function
    @inbounds for j=1:n
        gg[j] = 1.0 - X.data[j,j]
    end

    w, inds = proj.w, 1:proj.m[]
    return sum(y) + 0.5*Ldτ*dot(w,inds,w,inds)
end


function ncm(U::AbstractArray{T,2}, H::AbstractArray{T,2},
        proj::ProjPSD, storage::NCMstorage;
        method=:IAPG,
        exact=false,
        τ=1.0,
        α=0.0,
        σ=1.0,
        tol=1e-2,
        kmax=2000,
        f_calls_limit=2000,
        verbose=false,
        innerverbose=false,
        lbfgsbprintlevel=-1,
        cleanvals=true,
    ) where T

    # Loss function and gradient
    #f(X) = 0.5*norm(H.*(X .- U))^2
    #∇f(X) = Symmetric(H2.*(X .- U))

    # Check for valid input
    n = size(U, 1)
    n==storage.n || error("require n == storage.n")
    f_calls_limit ≤ storage.f_calls_limit ||
        error("require f_calls_limit ≤ storage.f_calls_limit")
    size(U)==size(H) || error("U and H must be the same size")
    issymmetric(U) || error("U must be symmetric")
    issymmetric(H) || error("H must be symmetric")
    !iszero(H) || error("H must be nonzero")

    method in NCMmethods || error("method must be in $NCMmethods")
    verbose && println("$method method")

    g = storage.g
    d = storage.d
    M = storage.M
    H2 = storage.H2
    Y = storage.Y
    ∇fY = storage.∇fY
    Γ = storage.Γ
    V = storage.V
    X = storage.X
    Xold = storage.Xold
    Z = storage.Z

    memlim = storage.memlim
    wa = storage.wa
    iwa = storage.iwa
    nbd = storage.nbd
    lower = storage.lower
    upper = storage.upper
    task = storage.task
    task2 = storage.task2
    csave = storage.csave
    lsave = storage.lsave
    isave = storage.isave
    dsave = storage.dsave

    nRef = storage.nRef
    mRef = storage.mRef
    iprint = storage.iprint
    fRef = storage.fRef
    factr = storage.factr
    pgtol = storage.pgtol

    iprint[] = lbfgsbprintlevel

    rpRef   = storage.rpRef
    rdRef   = storage.rdRef
    εRef    = storage.εRef
    δRef    = storage.δRef
    βRef    = storage.βRef
    distRef = storage.distRef

    res = storage.res
    Xnew = res.X
    y = res.y
    Λ = res.Λ
    fgcount = res.fgcount
    fvals   = res.fvals
    resvals = res.resvals

    fill!(y, 0)
    fill!(g, 0)
    fill!(d, 0)
    fill!(M, 0)
    fill!(H2, 0)
    fill!(Y, 0)
    fill!(∇fY, 0)
    fill!(Λ, 0)
    fill!(Γ, 0)
    fill!(V, 0)
    fill!(X, 0)
    fill!(Xold, 0)
    fill!(Z, 0)
    fill!(Xnew, 0)

    H2.data .= H.^2

    # Lipschitz constant of ∇f
    L = fronorm(H2, proj.work)

    if method==:IAPG
        τ==1 || error("IAPG method requires τ = 1")
        t0 = 1.0
    end

    if method==:IR
        0 < τ ≤ 1 || error("IR method requires 0 < τ ≤ 1")
        0 ≤ α ≤ (1 - τ)*(L/τ) || error("IR method requires 0 ≤ α ≤ $((1 - τ)*(L/τ))")
        t0 = 1.0
    end

    if method==:IER
        τ==1 || error("IER method requires τ = 1")
        α > 1/L || error("IER method requires α > $(1/L)")
        0 ≤ σ ≤ 1 || error("IER method requires 0 ≤ σ ≤ 1")
        λ = α/(1 + α*L)
        t0 = 0.0
    end

    k = 0
    t = t0
    gtol = NaN
    rp = rd = Inf
    fgcount[] = 0
    innersuccess = true

    while ( innersuccess &&
            max(rp, rd) > tol &&
            k < kmax &&
            fgcount[] < f_calls_limit )

        k += 1

        if method==:IER
            tnew = t + (λ + √(λ^2 + 4λ*t))/2
            Y.data .= (t/tnew).*Xnew.data .+ ((tnew - t)/tnew).*Xold.data
        else
            tnew = (1 + √(1 + 4t^2))/2
        end

        # Reset L-BFGS-B arrays
        fill!(wa, 0.0)
        fill!(iwa, 0)
        fill!(task, Cuchar(' '))
        fill!(csave, Cuchar(' '))
        fill!(lsave, 0)
        fill!(isave, 0)
        fill!(dsave, 0.0)

        maxfgcalls = f_calls_limit - fgcount[]

        if method==:IAPG
            gtol = (1 + √n)*min(1/tnew^3.1, 0.2*rd)
        end

        if exact
            gtol = 0.0
        end

        # Solve the subproblem
        innersuccess = calllbfgsb!(g, y, proj,
            H, H2, Y, U, ∇fY, M, X, Λ, Γ, d, Xnew, V, Z,
            fgcount, fvals, resvals,
            rpRef, rdRef, εRef, δRef, βRef, distRef,
            L, τ, α, σ,
            n, memlim, wa, iwa, nbd, lower, upper, task, task2, csave, lsave, isave, dsave,
            nRef, mRef, iprint, fRef, factr, pgtol;
            method=method,
            maxfgcalls=maxfgcalls,
            gtol=gtol,
            exact=exact,
            verbose=innerverbose,
            cleanvals=cleanvals,
        )
        if !innersuccess
            println("Failed to solve subproblem.")
        end
        fgcalls = fgcount[] - (f_calls_limit - maxfgcalls)

        fval = fvals[fgcount[]]
        rp   = rpRef[]
        rd   = rdRef[]
        ε    = εRef[]
        δ    = δRef[]
        β    = βRef[]
        dist = distRef[]

        if verbose
            mod(k, 20)==1 &&
            @printf("%4s %8s %10s %10s %10s %10s %10s %10s %10s %10s\n",
                "k", "fgcalls", "||g||", "gtol", "f(X)", "rp", "rd", "<X,Λ>", "||V||", "||X-Y||")
            @printf("%4d %8d %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e\n",
                k, fgcalls, norm(g), gtol, fval, rp, rd, ε, δ, dist)
        end

        if method==:IR
            ε = max(0.0, ε)
            condition = ((τ*δ)^2 + 2τ*ε*L ≤ L*((1-τ)*L - α*τ)*dist^2)
            if !condition && (fgcount[] < f_calls_limit)
                verbose && println("WARNING: (τ*δ)^2 + 2τ*ε*L ≤ L*((1-τ)*L - α*τ)*dist^2 fails")
            end
        end

        if method==:IER
            ε = max(0.0, ε)
            condition = (β^2 + 2α*ε ≤ (σ*dist)^2)
            if !condition && (fgcount[] < f_calls_limit)
                verbose && println("WARNING: β^2 + 2α*ε ≤ (σ*dist)^2 fails")
            end
        end

        # Update
        if method==:IAPG
            Y.data .= Xnew.data .+ ((t - 1)/tnew).*(Xnew.data .- Xold.data)
            Xold .= Xnew
        end

        if method==:IR
            Y.data .= Xnew.data .- ((t/tnew)*(τ/L)).*V.data .+ ((t - 1)/tnew).*(Xnew.data .- Xold.data)
            Xold .= Xnew
        end

        if method==:IER
            Xold.data .-= (tnew - t).*(V.data .+ L.*(Y.data .- Xnew.data))
        end

        t = tnew
    end

    if max(rp, rd) > tol
        verbose && println("Failed to converge after $(fgcount[]) function evaluations.")
    else
        verbose && println("Converged after $(fgcount[]) function evaluations.")
    end

    return res
end

