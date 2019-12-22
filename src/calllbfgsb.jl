using Printf, LBFGSB

const START = b"START"
const FG = b"FG"
const STOP = b"STOP"
const NEW_X = b"NEW_X"

function calllbfgsb!(func!, g, y, 
        H2, Y, U, ∇fY, M, X, Λ, Γ, d, Xnew, V, 
        fvals, resvals, rpvals, rdvals, L, τ, α, σ,
        n, memlim, wa, iwa, nbd, lower, upper, task, csave, lsave, isave, dsave;
        method=:IAPG,
        maxfgcalls=100,
        gtol=1e-2,
        exact=false,
        iprint=-1,
        verbose=false,
    )

    nRef = Ref{Cint}(n)
    mRef = Ref{Cint}(memlim)

    iprint = Ref{Cint}(iprint)    # print output at every iteration

    # "f is a DOUBLE PRECISION variable.
    # If the routine setulb returns with task(1:2)= 'FG', then f must be
    # set by the user to contain the value of the function at the point x."
    # "g is a DOUBLE PRECISION array of length n.
    # If the routine setulb returns with taskb(1:2)= 'FG', then g must be
    # set by the user to contain the components of the gradient at the
    # point x."
    fRef = Ref{Cdouble}(0.0)
    
    # specify the tolerances in the stopping criteria
    factr = Ref{Cdouble}(0.0)
    pgtol = Ref{Cdouble}(0.0)

    fgcalls = 0
    linesearchcalls = [0]
    sizehint!(linesearchcalls, 8)
    
    StopBFGS = false
    successful = true

    # "We start the iteration by initializing task."
    copyto!(task, START)

    while !StopBFGS

        # This is the call to the L-BFGS-B code.
        setulb(nRef, mRef, y, lower, upper, nbd,
            fRef, g, factr, pgtol, wa, iwa, task,
            iprint, csave, lsave, isave, dsave)

        if view(task, 1:2) == FG
            if fgcalls >= maxfgcalls
                copyto!(task, STOP)
            else
                fRef[] = func!(g, y, 
                    H2, Y, U, ∇fY, M, X, Λ, Γ, d, Xnew, V, 
                    fvals, resvals, rpvals, rdvals, L, τ)
                fgcalls += 1
                linesearchcalls[end] += 1
                    
                if verbose
                    if fgcalls == 1
                        @printf("\n%8s %10s %10s %10s", 
                            "fgcalls", "fRef[]", "norm(g)", "gtol")
                    end
                    @printf("\n%8d %10.2e %10.2e %10.2e", 
                        fgcalls, fRef[], norm(g), gtol)
                end
                
                if !exact                
                    if method==:IAPG
                        condition = (norm(g) < gtol)
                    else
                        ε = max(0.0, dot(Xnew, Λ))
                        δ = norm(V)
                        dist = norm(M.data .= Xnew .- Y)
                        if method==:IR
                            condition = ((τ*δ)^2 + 2τ*ε*L ≤ L*((1-τ)*L - α*τ)*dist^2)
                            verbose && @printf(" %10.2e, %10.2e", (τ*δ)^2 + 2τ*ε*L, L*((1-τ)*L - α*τ)*dist^2)
                        else # method==:IER
                            M.data .+= α.*V
                            β = norm(M)
                            condition = (β^2 + 2α*ε ≤ (σ*dist)^2)
                            verbose && @printf(" %10.2e, %10.2e", β^2 + 2α*ε, (σ*dist)^2)
                        end
                    end

                    if condition
                        copyto!(task, STOP)
                    end
                end
            end
            
            if fgcalls == 1 && view(task, 1:4) != STOP
                push!(linesearchcalls, 0)
            end

        elseif view(task, 1:5) == NEW_X
            verbose && @printf(" (linesearch complete)")
            push!(linesearchcalls, 0)
            
        else
            StopBFGS = true
            if !exact && view(task, 1:4) != STOP
                @printf("\ntask = %s\n", String(copy(task)))
                successful = false
            end
        end
    end
    verbose && @printf("\n")
    
    return successful, linesearchcalls
end
