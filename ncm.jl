struct NCMstorage
    n::Int
    memlim::Int
    y::Vector{Float64}
    g::Vector{Float64}
    d::Vector{Float64}
    M::Symmetric{Float64,Array{Float64,2}}
    H2::Symmetric{Float64,Array{Float64,2}}
    Y::Symmetric{Float64,Array{Float64,2}}
    ∇fY::Symmetric{Float64,Array{Float64,2}}
    Λ::Symmetric{Float64,Array{Float64,2}}
    Γ::Symmetric{Float64,Array{Float64,2}}
    V::Symmetric{Float64,Array{Float64,2}}
    X::Symmetric{Float64,Array{Float64,2}}
    Xold::Symmetric{Float64,Array{Float64,2}}
    Xnew::Symmetric{Float64,Array{Float64,2}}
    wa::Vector{Float64}
    iwa::Vector{Int32}
    nbd::Vector{Int32}
    lower::Vector{Float64}
    upper::Vector{Float64}
    task::Vector{UInt8}
    csave::Vector{UInt8}
    lsave::Vector{Int32}
    isave::Vector{Int32}
    dsave::Vector{Float64}

    function NCMstorage(n, memlim)
        y = zeros(n)
        g = zeros(n)
        d = zeros(n)

        M = Symmetric(zeros(n,n))
        H2 = copy(M)
        Y = copy(M)
        ∇fY = copy(M)
        Λ = copy(M)
        Γ = copy(M)
        V = copy(M)
        X = copy(M)
        Xold = copy(M)
        Xnew = copy(M)

        nmax = n
        mmax = memlim

        wa = zeros(Cdouble, 2mmax*nmax + 5nmax + 11mmax*mmax + 8mmax)
        iwa = zeros(Cint, 3nmax)

        # provide nbd which defines the bounds on the variables:
        nbd = zeros(Cint, nmax)         # no bounds on the variables
        lower = zeros(Cdouble, nmax)    # the lower bounds
        upper = zeros(Cdouble, nmax)    # the upper bounds

        task  = fill(Cuchar(' '), 60)   # fortran's blank padding
        csave = fill(Cuchar(' '), 60)   # fortran's blank padding
        lsave = zeros(Cint, 4)
        isave = zeros(Cint, 44)
        dsave = zeros(Cdouble, 29)

        new(n, memlim, y, g, d, M, H2, Y, ∇fY, Λ, Γ, V, X, Xold, Xnew, 
            wa, iwa, nbd, lower, upper, task, csave, lsave, isave, dsave)
    end
end


function ncm(U::AbstractArray{T,2}, H::AbstractArray{T,2}, myproj::ProjPSD, storage::NCMstorage; 
        method=:IAPG, 
        τ=1.0,
        α=0.0,
        σ=1.0,
        tol=1e-2,
        kmax=2000, 
        f_calls_limit=2000, 
        verbose=false,
        lbfgsbverbose=false,
        cleanvals=true,
    ) where T
    
    # Loss function and gradient
    #f(X) = 0.5*norm(H.*(X .- U))^2
    #∇f(X) = Symmetric(H2.*(X .- U))
    
    y = storage.y
    g = storage.g
    d = storage.d
    M = storage.M
    H2 = storage.H2
    Y = storage.Y
    ∇fY = storage.∇fY
    Λ = storage.Λ
    Γ = storage.Γ
    V = storage.V
    X = storage.X
    Xold = storage.Xold
    Xnew = storage.Xnew
    
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
    fill!(Xnew, 0)
    
    memlim = storage.memlim
    wa = storage.wa
    iwa = storage.iwa
    nbd = storage.nbd
    lower = storage.lower
    upper = storage.upper
    task = storage.task
    csave = storage.csave
    lsave = storage.lsave
    isave = storage.isave
    dsave = storage.dsave
        
    # Check for valid input
    method in [:IAPG, :IR, :IER] || error("method must be :IAPG, :IR, or :IER")
    println("$method method")

    issymmetric(U) || error("U must be symmetric")
    issymmetric(H) || error("H must be symmetric")
    size(U)==size(H) || error("U and H must be the same size")
    
    n = size(U, 1)
    n==storage.n || error("n != storage.n")
        
    H2.data .= H.*H
    
    # Lipschitz constant of ∇f
    L = norm(H2)
    
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
    
    # Evaluates dual objective function and its gradient
    function dualobj!(gg, y, 
            H2, Y, U, ∇fY, M, X, Λ, Γ, d, Xnew, V, 
            fvals, resvals, Rpvals, Rdvals, L, τ)

        ∇fY.data .= H2.*(Y .- U)
        M .= ∇fY; plusdiag!(M, y)  # M = ∇f(Y) + Diag(y)
        M.data .= Y .- (τ/L).*M    # M = Y - (τ/L)*(∇f(Y) + Diag(y))
        X .= M
        myproj(X)

        # Update Λ and Γ
        Λ.data .= (L/τ).*(X .- M)       # Λ is psd
        Γ.data .= .-Λ; plusdiag!(Γ, y)  # Γ = Diag(y) - Λ

        # Ensure that diag(Xnew).==1 exactly
        if isdiagallpos(X)
            for i=1:n
                @inbounds d[i] = 1/sqrt(X[i,i])
            end
            for j=1:n
                for i=1:n
                    @inbounds Xnew.data[i,j] = d[i]*d[j]*X[i,j]
                end
            end
        end

        # Update V
        V.data .= ∇fY .+ (L/τ).*(Xnew .- Y) .+ Γ

        # Compute and store the optim. cond. residual
        for i=1:n
            @inbounds d[i] = 1 - Xnew[i,i]
        end
        Rp = norm(d)/(1 + √n)
        Rd = norm(M.data .= H2.*(Xnew .- U) .+ Γ)
        push!(Rpvals, Rp)
        push!(Rdvals, Rd)
        push!(resvals, max(Rp,Rd))

        # Compute and store the objective function
        M.data .= H.*(Xnew .- U)
        push!(fvals, 0.5*dot(M,M))

        # Compute the gradient of the dual function
        for i=1:n
            @inbounds gg[i] = 1 - X[i,i]
        end

        w = view(myproj.w, 1:myproj.m[])
        return sum(y) + (L/2τ)*(dot(w,w) - dot(Y,Y))
    end
    
    k = 0
    t = t0
    gtol = NaN
    Rp = Rd = Inf
    innersuccess = true
    fvals   = Float64[]
    resvals = Float64[]
    Rpvals  = Float64[]
    Rdvals  = Float64[]
    
    while ( #innersuccess && 
            max(Rp, Rd) > tol && 
            k < kmax && 
            length(fvals) < f_calls_limit )
        
        k += 1
        
        if method==:IER
            tnew = t + (λ + √(λ^2 + 4λ*t))/2
            Y.data .= (t/tnew).*Xnew .+ ((tnew - t)/tnew).*Xold
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
            
        maxfgcalls = f_calls_limit - length(fvals)

        if method==:IAPG
            gtol = (1 + √n)*min(1/tnew^3.1, 0.2*Rd)
        end
        
        # Solve the subproblem
        innersuccess, linesearchcalls = calllbfgsb!(dualobj!, g, y, 
            H2, Y, U, ∇fY, M, X, Λ, Γ, d, Xnew, V, fvals, resvals, Rpvals, Rdvals, L, τ, α, σ,
            n, memlim, wa, iwa, nbd, lower, upper, task, csave, lsave, isave, dsave,
            method=method,
            maxfgcalls=maxfgcalls,
            gtol=gtol,
            verbose=lbfgsbverbose,
        )
        innersuccess || println("Failed to solve subproblem.")
        fgcalls = sum(linesearchcalls)

        if cleanvals
            cleanvals!(fvals, linesearchcalls)
            cleanvals!(resvals, linesearchcalls)
        end

        Rp = Rpvals[end]
        Rd = Rdvals[end]
        ε = dot(Xnew, Λ)
        δ = norm(V)
        dist = norm(M.data .= Xnew .- Y)

        if verbose
            mod(k, 20)==1 &&
            @printf("%4s %8s %10s %10s %14s %10s %10s %10s %10s %10s\n", 
                "k", "fgcalls", "||g||", "gtol", "f(X)", "Rp", "Rd", "<X,Λ>", "||V||", "||X-Y||")
            @printf("%4d %8d %10.2e %10.2e %14.6e %10.2e %10.2e %10.2e %10.2e %10.2e\n", 
                k, fgcalls, norm(g), gtol, fvals[end], Rp, Rd, ε, δ, dist)
        end
        
        if method==:IR
            ε = max(0.0, ε)
            condition = ((τ*δ)^2 + 2τ*ε*L ≤ L*((1-τ)*L - α*τ)*dist^2)
            if !condition && (length(fvals) < f_calls_limit)
                println("WARNING: (τ*δ)^2 + 2τ*ε*L ≤ L*((1-τ)*L - α*τ)*dist^2 fails")
                #@show (τ*δ)^2 + 2τ*ε*L, L*((1-τ)*L - α*τ)*dist^2
            end
        end
        
        if method==:IER
            ε = max(0.0, ε)
            M.data .+= α.*V
            β = norm(M)
            condition = (β^2 + 2α*ε ≤ (σ*dist)^2)
            if !condition && (length(fvals) < f_calls_limit)
                println("WARNING: β^2 + 2α*ε ≤ (σ*dist)^2 fails")
                #@show β^2 + 2α*ε, (σ*dist)^2
            end
        end

        # Update
        if method==:IAPG
            Y.data .= Xnew .+ ((t - 1)/tnew).*(Xnew .- Xold)
            Xold .= Xnew
        end
        
        if method==:IR
            Y.data .= Xnew .- ((t/tnew)*(τ/L)).*V .+ ((t - 1)/tnew).*(Xnew .- Xold)
            Xold .= Xnew
        end
        
        if method==:IER
            Xold.data .-= (tnew - t).*(V .+ L.*(Y .- Xnew))
        end
        
        t = tnew
    end
    
    if max(Rp, Rd) > tol
        println("Failed to converge after $(length(fvals)) function evaluations.")
    else
        println("Converged after $(length(fvals)) function evaluations.")
    end
    
    return Xnew, y, fvals, resvals
end
