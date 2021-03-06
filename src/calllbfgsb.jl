using Printf, LBFGSB

const START = b"START"
const FG = b"FG"
const STOP = b"STOP"
const NEW_X = b"NEW_X"

const STOP2 = b"ST"
const NEW_X2 = b"NE"

function copyval!(x, copytoinds, copyfromind)
    for i in copytoinds
        @inbounds x[i] = x[copyfromind]
    end
end

function calllbfgsb!(
    ncm,
    G,
    H,
    tol,
    t,
    L,
    τ,
    α,
    σ;
    method::Symbol = :IAPG,
    innertol::Float64 = 1e-2,
    maxfgcalls::Int64 = 100,
    lbfgsbprintlevel::Int64 = -1,
    exact::Bool = false,
    verbose::Bool = false,
    cleanvals::Bool = true,
    scaleX::Bool = true,
)

    n = ncm.n
    g = ncm.g

    V = ncm.V
    Z = ncm.Z
    εRef = ncm.εRef

    wa = ncm.wa
    iwa = ncm.iwa
    nbd = ncm.nbd
    lower = ncm.lower
    upper = ncm.upper
    task = ncm.task
    task2 = ncm.task2
    csave = ncm.csave
    lsave = ncm.lsave
    isave = ncm.isave
    dsave = ncm.dsave
    nRef = ncm.nRef
    mRef = ncm.mRef
    iprint = ncm.iprint
    fRef = ncm.fRef
    factr = ncm.factr
    pgtol = ncm.pgtol

    iprint[] = lbfgsbprintlevel

    proj = ncm.proj
    res = ncm.res

    Xnew = res.X
    y = res.y
    Λ = res.Λ

    fvals = res.fvals
    resvals = res.resvals
    distvals = res.distvals
    fgcountRef = res.fgcountRef

    # Reset L-BFGS-B arrays
    fill!(wa, 0.0)
    fill!(iwa, 0)
    fill!(task, Cuchar(' '))
    fill!(csave, Cuchar(' '))
    fill!(lsave, 0)
    fill!(isave, 0)
    fill!(dsave, 0.0)

    fgcalls = 0
    linesearchcount = 0

    StopBFGS = false
    successful = true
    computeV = (method == :IAPG || method == :IR || method == :IER)

    # "We start the iteration by initializing task."
    copyto!(task, START)

    while !StopBFGS

        # This is the call to the L-BFGS-B code.
        setulb(
            nRef,
            mRef,
            y,
            lower,
            upper,
            nbd,
            fRef,
            g,
            factr,
            pgtol,
            wa,
            iwa,
            task,
            iprint,
            csave,
            lsave,
            isave,
            dsave,
        )

        if cleanvals && linesearchcount > 1
            a = fgcountRef[] - linesearchcount + 1
            b = fgcountRef[]
            copytoinds = a:b-1
            copyval!(fvals, copytoinds, b)
            copyval!(resvals, copytoinds, b)
            copyval!(distvals, copytoinds, b)
        end

        if task2 == FG
            if fgcalls >= maxfgcalls
                copyto!(task, STOP)
            else
                fRef[] = dualobj!(
                    ncm,
                    G,
                    H,
                    L,
                    τ,
                    computeV = computeV,
                    scaleX = scaleX,
                )
                fgcalls += 1
                if fgcalls > 1
                    linesearchcount += 1
                end

                if resvals[fgcountRef[]] < tol
                    copyto!(task, STOP)
                elseif !exact
                    if method == :IAPG
                        δ = fronorm(V, proj.work)
                        lhs = 1/√L * δ
                        rhs = innertol / (√2 * t)
                        condition = (lhs ≤ rhs)
                    else
                        εRef[] = symdot(Xnew, Λ)
                        ε = max(0.0, εRef[])
                        dist = distvals[fgcountRef[]]
                        if method == :IR
                            δ = fronorm(V, proj.work)
                            lhs = (τ * δ)^2 + 2τ * ε * L
                            rhs = L * ((1 - τ) * L - α * τ) * dist^2
                            condition = (lhs ≤ rhs)
                        elseif method == :IER
                            Z.data .+= α .* V.data
                            β = fronorm(Z, proj.work)
                            lhs = β^2 + 2α * ε
                            rhs = (σ * dist)^2
                            condition = (lhs ≤ rhs)
                        end
                    end

                    if condition
                        copyto!(task, STOP)
                    end
                end
            end

        elseif task2 == NEW_X2
            linesearchcount = 0

        else
            StopBFGS = true
            if !exact && task2 != STOP2
                verbose && @printf("\ntask = %s\n", String(copy(task)))
                successful = false
            end
        end

    end

    return successful
end
