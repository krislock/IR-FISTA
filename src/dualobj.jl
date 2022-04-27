# Evaluates dual objective function and its gradient
function dualobj!(ncm, G, H, L, τ; computeV::Bool = false, scaleX::Bool = true)

    n = ncm.n
    d = ncm.d
    g = ncm.g

    M = ncm.M
    R = ncm.R
    H2 = ncm.H2
    Y = ncm.Y
    ∇fY = ncm.∇fY
    Γ = ncm.Γ
    V = ncm.V
    X = ncm.X
    Z = ncm.Z
    Rd = ncm.Rd

    proj = ncm.proj
    res = ncm.res

    Xnew = res.X
    y = res.y
    Λ = res.Λ

    rpRef = res.rpRef
    rdRef = res.rdRef

    fvals = res.fvals
    resvals = res.resvals
    distvals = res.distvals

    res.fgcountRef[] += 1
    fgcount = res.fgcountRef[]

    τdL = τ / L
    Ldτ = L / τ

    # ∇fY = H2.*(Y - G)
    # M = Y - (τ/L)*(∇f(Y) - Diag(y))
    # X = M
    @inbounds for j = 1:n
        for i = 1:j
            ∇fY.data[i, j] = H2.data[i, j] * (Y.data[i, j] - G.data[i, j])
            M.data[i, j] = Y.data[i, j] - τdL * ∇fY.data[i, j]
        end
        M.data[j, j] += τdL * y[j]
    end

    proj(M, X, Λ)

    if scaleX
        # Ensure that diag(Xnew).==1 exactly
        @inbounds for j = 1:n
            if X.data[j, j] > 0.0
                d[j] = 1.0 / sqrt(X.data[j, j])
            else
                d[j] = 1.0
            end
        end
    end

    # Λ  = Ldτ*(X - M)        # Λ is psd
    # Γ  = -Diag(y) - Λ       # Γ is an ε-subgradient
    # Z  = Xnew - Y           # used to compute ||Xnew - Y||
    # V  = ∇f(Y) + Ldτ*Z + Γ  # used in the IR update
    # R  = H.*(Xnew - G)      # used to compute primal objective
    # Rd = H.*R + Γ           # used to compute dual feasibility

    @inbounds for j = 1:n
        for i = 1:j
            if scaleX
                Xnew.data[i, j] = d[i] * d[j] * X.data[i, j]
            else
                Xnew.data[i, j] = X.data[i, j]
            end
            #Λ.data[i,j] = Ldτ*(X.data[i,j] - M.data[i,j])
            Λ.data[i, j] *= Ldτ
            Γ.data[i, j] = -Λ.data[i, j]
            Z.data[i, j] = Xnew.data[i, j] - Y.data[i, j]
            if computeV
                #V.data[i, j] = ∇fY.data[i, j] + Ldτ * Z.data[i, j] + Γ.data[i, j]
                V.data[i,j] = Ldτ * (Xnew.data[i,j] - X.data[i,j])
            end
            R.data[i, j] = H.data[i, j] * (Xnew.data[i, j] - G.data[i, j])
            Rd.data[i, j] = H.data[i, j] * R.data[i, j] + Γ.data[i, j]
        end
        Γ.data[j, j] -= y[j]
        #if computeV
        #    V.data[j, j] -= y[j]
        #end
        Rd.data[j, j] -= y[j]
    end

    # Compute primal and dual feasibility
    @inbounds for j = 1:n
        d[j] = Xnew.data[j, j] - 1.0
    end
    rpRef[] = norm(d)
    rdRef[] = fronorm(Rd, proj.work)

    fvals[fgcount] = 0.5 * fronorm(R, proj.work)^2
    resvals[fgcount] = max(rpRef[], rdRef[])
    distvals[fgcount] = fronorm(Z, proj.work)

    # Compute the gradient of the dual function
    @inbounds for j = 1:n
        g[j] = X.data[j, j] - 1.0
    end

    # Compute the value of the dual function
    λ = proj.w
    dualobjval = 0.0
    @inbounds for j = 1:n
        tmp = λ[j]
        if tmp > 0.0
            dualobjval += tmp^2
        end
    end
    dualobjval *= 0.5 * Ldτ
    dualobjval -= sum(y)

    return dualobjval
end
