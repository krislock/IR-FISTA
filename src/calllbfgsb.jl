using Printf, LBFGSB

const START = b"START"
const FG    = b"FG"
const STOP  = b"STOP"
const NEW_X = b"NEW_X"

const STOP2  = b"ST"
const NEW_X2 = b"NE"

function copyval!(x, copytoinds, copyfromind)
    for i = copytoinds
        @inbounds x[i] = x[copyfromind]
    end
end

function calllbfgsb!(g, y, proj,
        H, H2, Y, U, ∇fY, M, X, Λ, Γ, d, Xnew, V, Z,
        fgcount, fvals, resvals,
        rpRef, rdRef, εRef, δRef, βRef, distRef,
        L, τ, α, σ,
        n, memlim, wa, iwa, nbd, lower, upper,
        task, task2, csave, lsave, isave, dsave,
        nRef, mRef, iprint, fRef, factr, pgtol;
        method=:IAPG,
        maxfgcalls=100,
        gtol=1e-2,
        exact=false,
        verbose=false,
        cleanvals=true,
    )

    fgcalls = 0
    linesearchcount = 0

    StopBFGS = false
    successful = true

    # "We start the iteration by initializing task."
    copyto!(task, START)

    while !StopBFGS

        # This is the call to the L-BFGS-B code.
        setulb(nRef, mRef, y, lower, upper, nbd,
            fRef, g, factr, pgtol, wa, iwa, task,
            iprint, csave, lsave, isave, dsave)

        if cleanvals && linesearchcount > 1
            a, b = fgcount[]-linesearchcount+1, fgcount[]
            copytoinds = a:b-1
            copyval!(fvals,   copytoinds, b)
            copyval!(resvals, copytoinds, b)
        end

        if task2 == FG
            if fgcalls >= maxfgcalls
                copyto!(task, STOP)
            else
                fRef[] = dualobj!(g, y, proj, method,
                    n, H, H2, Y, U, ∇fY, M, X, Λ, Γ, d, Xnew, V, Z,
                    fgcount, fvals, resvals,
                    rpRef, rdRef, εRef, δRef, distRef, L, τ)

                fgcalls += 1
                if fgcalls > 1
                    linesearchcount += 1
                end

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
                        εRef[]    = symdot(Xnew, Λ)
                        distRef[] = fronorm(Z, proj.work)
                        ε = max(0.0, εRef[])
                        dist = distRef[]
                        if method==:IR
                            δRef[] = fronorm(V, proj.work)
                            δ = δRef[]
                            condition = ((τ*δ)^2 + 2τ*ε*L ≤ L*((1-τ)*L - α*τ)*dist^2)
                        else # method==:IER
                            Z.data .+= α.*V.data
                            βRef[] = fronorm(Z, proj.work)
                            β = βRef[]
                            condition = (β^2 + 2α*ε ≤ (σ*dist)^2)
                        end
                    end

                    if condition
                        copyto!(task, STOP)
                    end
                end
            end

        elseif task2 == NEW_X2
            verbose && @printf(" (linesearch complete)")
            linesearchcount = 0

        else
            StopBFGS = true
            if !exact && task2 != STOP2
                verbose && @printf("\ntask = %s\n", String(copy(task)))
                successful = false
            end
        end

    end
    verbose && @printf("\n")

    return successful
end

