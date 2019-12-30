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
            Rd.data[i,j] = H.data[i,j]*R.data[i,j] + Γ.data[i,j]
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
    dualobjval = sum(y) + 0.5*Ldτ*dot(w,inds,w,inds)

    return dualobjval
end
