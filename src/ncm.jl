#using JuMP, CSDP

function optimalityconditions(U, H, X, y, Λ)
    rp = norm(1 .- diag(X))
    rd = norm(H.*H.*(X .- U) .+ Diagonal(y) .- Λ)
    @show rp
    @show rd
    @show dot(X, Λ)
    @show minimum(eigvals(X))
    @show minimum(eigvals(Λ))
    return nothing
end

#=
function ncm(U, H; verbose=false)
    # Initialize model with correlation matrix constraints
    m = Model(with_optimizer(CSDP.Optimizer))
    if !verbose
        set_silent(m)
    end
    @variable(m, X[1:n,1:n], Symmetric)
    @constraint(m, diagcon, diag(X) .== 1)
    @constraint(m, psdcon, X in PSDCone())
    R = H.*(X .- U)
    @objective(m, Min, 0.5*dot(R, R))
    optimize!(m)
    Xval = value.(X)
    y = dual.(diagcon)
    Λ = dual.(psdcon)
    return Xval, y, Λ
end
=#


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


struct NCMresults
    X::Symmetric{Float64,Array{Float64,2}}
    y::Vector{Float64}
    Λ::Symmetric{Float64,Array{Float64,2}}
    fvals::Vector{Float64}
    resvals::Vector{Float64}
end


function ncm(U::AbstractArray{T,2}, H::AbstractArray{T,2}, myproj::ProjPSD, storage::NCMstorage; 
        method=:IAPG, 
        exact=false,
        τ=1.0,
        α=0.0,
        σ=1.0,
        tol=1e-2,
        kmax=4000, 
        f_calls_limit=8000, 
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
    verbose && println("$method method")

    issymmetric(U) || error("U must be symmetric")
    issymmetric(H) || error("H must be symmetric")
    size(U)==size(H) || error("U and H must be the same size")
    any(hij != 0.0 for hij in H) || error("H must be nonzero")
    
    n = size(U, 1)
    n==storage.n || error("n != storage.n")
        
    H2.data .= H.^2
    
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
            fgcount, fvals, resvals, rpvals, rdvals, L, τ)

        fgcount[1] += 1

        τdL = τ/L
        Ldτ = L/τ

        #=
        ∇fY.data .= H2.*(Y .- U)
        M .= ∇fY; plusdiag!(M, y)  # M = ∇f(Y) + Diag(y)
        M.data .= Y .- τdL.*M      # M = Y - (τ/L)*(∇f(Y) + Diag(y))
        X .= M
        =#
        @inbounds for j=1:n
            for i=1:j
                ∇fY.data[i,j] = H2[i,j]*(Y[i,j] - U[i,j])
                M.data[i,j] = Y[i,j] - τdL*∇fY[i,j]
                X.data[i,j] = M[i,j]
            end
            M.data[j,j] -= τdL*y[j]
            X.data[j,j] = M[j,j]
        end
        myproj(X)

        # Update Λ and Γ
        #Λ.data .= Ldτ.*(X .- M)         # Λ is psd
        #Γ.data .= .-Λ; plusdiag!(Γ, y)  # Γ = Diag(y) - Λ

        # Ensure that diag(Xnew).==1 exactly
        @inbounds for j=1:n
            if X[j,j] > 0.0
                d[j] = 1.0/sqrt(X[j,j])
            else
                d[j] = 1.0
            end
        end
        @inbounds for j=1:n
            for i=1:j
                Λ.data[i,j] = Ldτ*(X[i,j] - M[i,j])
                Γ.data[i,j] = -Λ[i,j]
                Xnew.data[i,j] = d[i]*d[j]*X[i,j]
                V.data[i,j] = ∇fY[i,j] + Ldτ*(Xnew[i,j] - Y[i,j]) + Γ[i,j]
                M.data[i,j] = H[i,j]*(Xnew[i,j] - U[i,j])
            end
            Γ.data[j,j] += y[j]
            V.data[j,j] += y[j]
        end

        # Update V
        # V.data .= ∇fY .+ Ldτ.*(Xnew .- Y) .+ Γ

        # Compute and store the objective function
        #M.data .= H.*(Xnew .- U)
        fvals[fgcount[1]] = 0.5*dot(M,M)

        # Compute and store the optim. cond. residual
        @inbounds for j=1:n
            d[j] = 1.0 - Xnew[j,j]
            for i=1:j
                M.data[i,j] = H[i,j]*M[i,j] + Γ[i,j]
            end
        end
        rpvals[fgcount[1]] = norm(d)/(1 + √n)
        #M.data .= H2.*(Xnew .- U) .+ Γ
        rdvals[fgcount[1]] = norm(M)
        resvals[fgcount[1]] = max(rpvals[fgcount[1]],rdvals[fgcount[1]])

        # Compute the gradient of the dual function
        @inbounds for j=1:n
            gg[j] = 1.0 - X[j,j]
        end

        w = view(myproj.w, 1:myproj.m[])
        return sum(y) + 0.5*Ldτ*dot(w,w)
    end
    
    k = 0
    t = t0
    gtol = NaN
    rp = rd = Inf
    innersuccess = true

    fgcount = [0]
    fvals   = Vector{Float64}(undef, f_calls_limit)
    resvals = Vector{Float64}(undef, f_calls_limit)
    rpvals  = Vector{Float64}(undef, f_calls_limit)
    rdvals  = Vector{Float64}(undef, f_calls_limit)
    
    while ( #innersuccess && 
            max(rp, rd) > tol && 
            k < kmax && 
            fgcount[1] < f_calls_limit )
        
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
            
        maxfgcalls = f_calls_limit - fgcount[1]

        if method==:IAPG
            gtol = (1 + √n)*min(1/tnew^3.1, 0.2*rd)
        end

        if exact
            gtol = 0.0
        end
        
        # Solve the subproblem
        innersuccess = calllbfgsb!(dualobj!, g, y,
            H2, Y, U, ∇fY, M, X, Λ, Γ, d, Xnew, V,
            fgcount, fvals, resvals, rpvals, rdvals, L, τ, α, σ,
            n, memlim, wa, iwa, nbd, lower, upper, task, csave, lsave, isave, dsave,
            method=method,
            maxfgcalls=maxfgcalls,
            gtol=gtol,
            exact=exact,            
            verbose=lbfgsbverbose,
            cleanvals=cleanvals,
        )
        if !innersuccess
            verbose && println("Failed to solve subproblem.")
        end
        fgcalls = fgcount[1] - (f_calls_limit - maxfgcalls)

        rp = rpvals[fgcount[1]]
        rd = rdvals[fgcount[1]]
        ε = dot(Xnew, Λ)
        δ = norm(V)
        dist = norm(M.data .= Xnew .- Y)

        if verbose
            mod(k, 20)==1 &&
            @printf("%4s %8s %10s %10s %14s %10s %10s %10s %10s %10s\n", 
                "k", "fgcalls", "||g||", "gtol", "f(X)", "rp", "rd", "<X,Λ>", "||V||", "||X-Y||")
            @printf("%4d %8d %10.2e %10.2e %14.6e %10.2e %10.2e %10.2e %10.2e %10.2e\n", 
                k, fgcalls, norm(g), gtol, fvals[fgcount[1]], rp, rd, ε, δ, dist)
        end
        
        if method==:IR
            ε = max(0.0, ε)
            condition = ((τ*δ)^2 + 2τ*ε*L ≤ L*((1-τ)*L - α*τ)*dist^2)
            if !condition && (fgcount[1] < f_calls_limit)
                verbose && println("WARNING: (τ*δ)^2 + 2τ*ε*L ≤ L*((1-τ)*L - α*τ)*dist^2 fails")
            end
        end
        
        if method==:IER
            ε = max(0.0, ε)
            M.data .+= α.*V
            β = norm(M)
            condition = (β^2 + 2α*ε ≤ (σ*dist)^2)
            if !condition && (fgcount[1] < f_calls_limit)
                verbose && println("WARNING: β^2 + 2α*ε ≤ (σ*dist)^2 fails")
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
    
    if max(rp, rd) > tol
        verbose && println("Failed to converge after $(fgcount[1]) function evaluations.")
    else
        verbose && println("Converged after $(fgcount[1]) function evaluations.")
    end
    
    return NCMresults(Xnew, y, Λ, fvals[1:fgcount[1]], resvals[1:fgcount[1]])
end
