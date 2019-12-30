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

    function NCMresults(n, f_calls_limit)
        X = Symmetric(zeros(n,n))
        y = zeros(n)
        Λ = Symmetric(zeros(n,n))
        fgcountRef = Ref{Int32}(0)
        fvals    = Vector{Float64}(undef, f_calls_limit)
        resvals  = Vector{Float64}(undef, f_calls_limit)
        distvals = Vector{Float64}(undef, f_calls_limit)
        rpRef    = Ref{Float64}(0.0)
        rdRef    = Ref{Float64}(0.0)

        new(X, y, Λ,
            fgcountRef, fvals, resvals, distvals, rpRef, rdRef)
    end
end


struct NCM
    n::Int32
    memlim::Int32
    f_calls_limit::Int32
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

    function NCM(n; memlim=10, f_calls_limit=2000)
        g = zeros(n)
        d = zeros(n)

        M    = Symmetric(zeros(n,n))
        R    = copy(M)
        H2   = copy(M)
        Y    = copy(M)
        ∇fY  = copy(M)
        Γ    = copy(M)
        V    = copy(M)
        X    = copy(M)
        Z    = copy(M)
        Rd   = copy(M)
        Xold = copy(M)

        nmax = n
        mmax = memlim

        wa  = zeros(Cdouble, 2mmax*nmax + 5nmax + 11mmax*mmax + 8mmax)
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
        fRef   = Ref{Cdouble}(0.0)
        factr  = Ref{Cdouble}(0.0)
        pgtol  = Ref{Cdouble}(0.0)

        εRef   = Ref{Float64}(0.0)

        proj = ProjPSD(n)

        res = NCMresults(n, f_calls_limit)

        new(n, memlim, f_calls_limit,
            g, d, M, R, H2, Y, ∇fY, Γ, V, X, Z, Rd, Xold,
            wa, iwa, nbd, lower, upper,
            task, task2, csave, lsave, isave, dsave,
            nRef, mRef, iprint, fRef, factr, pgtol,
            εRef, proj, res)
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
           @inbounds A.data[j,i] = A.data[i,j]
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
function dualobj!(ncm, U, H, L, τ;
                  method=method,
                  scaleX=scaleX,
                 )

    n = ncm.n
    d = ncm.d
    g = ncm.g

    M   = ncm.M
    R   = ncm.R
    H2  = ncm.H2
    Y   = ncm.Y
    ∇fY = ncm.∇fY
    Γ   = ncm.Γ
    V   = ncm.V
    X   = ncm.X
    Z   = ncm.Z
    Rd  = ncm.Rd

    proj = ncm.proj
    res  = ncm.res

    Xnew = res.X
    y    = res.y
    Λ    = res.Λ

    rpRef = res.rpRef
    rdRef = res.rdRef

    fvals    = res.fvals
    resvals  = res.resvals
    distvals = res.distvals

    res.fgcountRef[] += 1
    fgcount = res.fgcountRef[]

    τdL = τ/L
    Ldτ = L/τ

    # ∇fY.data .= H2.*(Y .- U)
    # M .= ∇fY .+ Diagonal(y)    # M = ∇f(Y) + Diag(y)
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

    if scaleX
        # Ensure that diag(Xnew).==1 exactly
        @inbounds for j=1:n
            if X.data[j,j] > 0.0
                d[j] = 1.0/sqrt(X.data[j,j])
            else
                d[j] = 1.0
            end
        end
    end
    if method==:IR || method==:IER
        computeV = true
    else
        computeV = false
    end

    # Λ.data  .= Ldτ.*(X .- M)                   # Λ is psd
    # Γ.data  .= Diagonal(y) .- Λ
    # Z.data  .= Xnew .- Y
    # V.data  .= ∇fY .+ Ldτ.*(Xnew .- Y) .+ Γ
    # R.data  .= H.*(Xnew .- U)
    # Rd.data .= H2.*(Xnew .- U) .+ Γ

    @inbounds for j=1:n
        for i=1:j
            if scaleX
                Xnew.data[i,j] = d[i]*d[j]*X.data[i,j]
            else
                Xnew.data[i,j] = X.data[i,j]
            end
            Λ.data[i,j] = Ldτ*(X.data[i,j] - M.data[i,j])
            Γ.data[i,j] = -Λ.data[i,j]
            Z.data[i,j] = Xnew.data[i,j] - Y.data[i,j]
            if computeV
                V.data[i,j] = ∇fY.data[i,j] + Ldτ*Z.data[i,j] + Γ.data[i,j]
            end
            R.data[i,j]  = H.data[i,j]*(Xnew.data[i,j] - U.data[i,j])
            Rd.data[i,j] = H.data[i,j]*M.data[i,j] + Γ.data[i,j]
        end
        Γ.data[j,j] += y[j]
        if computeV
            V.data[j,j] += y[j]
        end
        Rd.data[j,j] += y[j]
    end

    # Compute and store the objective function
    fvals[fgcount] = 0.5*fronorm(R, proj.work)^2

    # Compute and store the optim. cond. residual
    @inbounds for j=1:n
        d[j] = 1.0 - Xnew[j,j]
    end
    rpRef[] = norm(d)/(1 + √n)
    rdRef[] = fronorm(Rd, proj.work)
    resvals[fgcount]  = max(rpRef[],rdRef[])
    distvals[fgcount] = fronorm(Z, proj.work)

    # Compute the gradient of the dual function
    @inbounds for j=1:n
        g[j] = 1.0 - X.data[j,j]
    end

    w, inds = proj.w, 1:proj.m[]
    return sum(y) + 0.5*Ldτ*dot(w,inds,w,inds)
end


function (ncm::NCM)(U::Symmetric{Float64,Array{Float64,2}},
                    H::Symmetric{Float64,Array{Float64,2}};
                    method=:IAPG,
                    exact=false,
                    τ=1.0,
                    α=0.0,
                    σ=1.0,
                    useXold=false,
                    tol=1e-2,
                    kmax=2000,
                    f_calls_limit=2000,
                    printlevel=0,
                    innerverbose=false,
                    lbfgsbprintlevel=-1,
                    cleanvals=true,
                    scaleX=true,
                   )

    # Loss function and gradient
    #f(X) = 0.5*norm(H.*(X .- U))^2
    #∇f(X) = Symmetric(H2.*(X .- U))

    # Check for valid input
    n = size(U, 1)
    f_calls_limit ≤ ncm.f_calls_limit ||
        error("require f_calls_limit ≤ ncm.f_calls_limit")
    n==ncm.n         || error("require n == ncm.n")
    size(U)==size(H) || error("U and H must be the same size")
    issymmetric(U)   || error("U must be symmetric")
    issymmetric(H)   || error("H must be symmetric")
    !iszero(H)       || error("H must be nonzero")

    validmethod = (method==:IAPG || method==:IR || method==:IER)
    validmethod || error("method must be :IAPG or :IR or :IER")
    printlevel≥1 && println("$method method, τ=$τ, α=$α, σ=$σ, tol=$tol")

    g    = ncm.g
    H2   = ncm.H2
    Y    = ncm.Y
    V    = ncm.V
    Xold = ncm.Xold

    proj = ncm.proj
    res  = ncm.res

    Xnew = res.X
    y    = res.y

    fgcountRef = res.fgcountRef
    rpRef      = res.rpRef
    rdRef      = res.rdRef
    fvals      = res.fvals

    H2.data .= H.^2

    # Lipschitz constant of ∇f
    L = fronorm(H2, proj.work)

    if method==:IAPG
        τ==1 || error("IAPG method requires τ = 1")
        t0 = 1.0
    end

    if method==:IR
        0 < τ ≤ 1 || error("IR method requires 0 < τ ≤ 1")
        0 ≤ α ≤ (1 - τ)*(L/τ) ||
            error("IR method requires 0 ≤ α ≤ $((1 - τ)*(L/τ))")
        t0 = 1.0
    end

    if method==:IER
        τ==1      || error("IER method requires τ = 1")
        α > 1/L   || error("IER method requires α > $(1/L)")
        0 ≤ σ ≤ 1 || error("IER method requires 0 ≤ σ ≤ 1")
        λ = α/(1 + α*L)
        t0 = 0.0
    end

    if !useXold
        fill!(Xold, 0.0)
    end

    fill!(y, 0.0)

    k = 0
    t = t0
    Y .= Xold
    gtol = NaN
    rp = rd = Inf
    innersuccess = true
    fgcountRef[] = 0
    fgcount = fgcountRef[]

    while ( #innersuccess &&
            max(rp, rd) > tol &&
            k < kmax &&
            fgcount < f_calls_limit )

        k += 1

        if method==:IER
            tnew = t + (λ + √(λ^2 + 4λ*t))/2
            Y.data .= (t/tnew).*Xnew.data .+ ((tnew - t)/tnew).*Xold.data
        else
            tnew = (1 + √(1 + 4t^2))/2
        end

        if method==:IAPG
            gtol = (1 + √n)*min(1/tnew^3.1, 0.2*rd)
        end

        if exact
            gtol = 0.0
        end

        maxfgcalls = f_calls_limit - fgcount

        # Solve the subproblem
        innersuccess = calllbfgsb!(ncm, U, H, tol, L, τ, α, σ;
            method=method,
            maxfgcalls=maxfgcalls,
            gtol=gtol,
            exact=exact,
            verbose=innerverbose,
            lbfgsbprintlevel=lbfgsbprintlevel,
            cleanvals=cleanvals,
            scaleX=scaleX,
        )
        if !innersuccess
            printlevel≥2 && println("Failed to solve subproblem.")
        end
        fgcount = fgcountRef[]
        fgcalls = fgcount - (f_calls_limit - maxfgcalls)

        rp = rpRef[]
        rd = rdRef[]
        rankX = proj.m[]

        if printlevel≥2
            mod(k, 20)==1 &&
            @printf("%4s %8s %10s %10s %10s %10s %10s %8s\n",
                "k", "fgcalls", "||g||", "gtol", "f(X)", "rp", "rd", "rank(X)")
            @printf("%4d %8d %10.2e %10.2e %10.2e %10.2e %10.2e %8d\n",
                k, fgcalls, norm(g), gtol, fvals[fgcount], rp, rd, rankX)
        end

        # Update
        if method==:IAPG
            Y.data .= Xnew.data .+ ((t - 1)/tnew).*(Xnew.data .- Xold.data)
            Xold .= Xnew
        elseif method==:IR
            Y.data .= Xnew.data .- ((t/tnew)*(τ/L)).*V.data .+ ((t - 1)/tnew).*(Xnew.data .- Xold.data)
            Xold .= Xnew
        elseif method==:IER
            Xold.data .-= (tnew - t).*V.data .+ ((tnew - t)*L).*(Y.data .- Xnew.data)
        end

        t = tnew
    end

    if max(rp, rd) > tol
        success = false
        printlevel≥1 && println("Failed to converge after $fgcount function evaluations.")
    else
        success = true
        printlevel≥1 && println("Converged after $fgcount function evaluations.")
    end

    return success, k
end

