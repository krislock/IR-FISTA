using LinearAlgebra

struct ProjPSD
    nmax::Int
    jobz::Char
    range::Char
    il::Int
    iu::Int
    abstol::Float64
    m::Base.RefValue{Int}
    w::Vector{Float64}
    Z::Matrix{Float64}
    ldz::Int
    isuppz::Vector{Int}
    work::Vector{Float64}
    lwork::Int
    iwork::Vector{Int}
    liwork::Int
    info::Base.RefValue{Int}

    function ProjPSD(nmax)
        n = nmax
        A = Symmetric(zeros(1,1))

        jobz = 'V'
        range = 'V'
        lda = n
        vl = 0.0
        vu = Inf
        il = 0
        iu = 0
        abstol = -1.0
        m = Ref{Int}(0)
        w = zeros(n)
        Z = zeros(n, n)
        ldz = n
        isuppz = zeros(Int, 2n)
        work   = zeros(1)
        iwork  = zeros(Int, 1)
        info   = Ref{Int}(0)

        # Perform an optimal workspace query
        ccall((:dsyevr_64_, "libopenblas64_"), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{Int},
                Ptr{Float64}, Ref{Int}, Ref{Float64}, Ref{Float64},
                Ref{Int}, Ref{Int}, Ref{Float64}, Ptr{Int},
                Ptr{Float64}, Ptr{Float64}, Ref{Int}, Ptr{Int},
                Ptr{Float64}, Ref{Int}, Ptr{Int}, Ref{Int},
                Ptr{Int}),
            jobz, range, A.uplo, n,
            A.data, lda, vl, vu,
            il, iu, abstol, m,
            w, Z, ldz, isuppz,
            work, -1, iwork, -1,
            info)

        lwork = Int(real(work[1]))
        liwork = iwork[1]

        resize!(work, lwork)
        resize!(iwork, liwork)

        new(nmax, jobz, range, il, iu, abstol, m, w, Z, ldz,
            isuppz, work, lwork, iwork, liwork, info)
    end
end


function (obj::ProjPSD)(A::Symmetric)
    lda, n = size(A)

    @assert lda == n
    @assert n <= obj.nmax

    vl = 0.0
    vu = Inf  # Tests show that vu = Inf is faster than
              # vu = min(norminf, normfro)
    abstol = -1.0

    #=
    vl = 1e-8
    norminf = ccall((:dlansy_64_, "libopenblas64_"), Cdouble,
        (Ref{UInt8}, Ref{UInt8}, Ref{Int}, Ptr{Float64}, Ref{Int},
        Ptr{Float64}), 'I', A.uplo, n, A.data, lda, obj.work)
    normfro = ccall((:dlansy_64_, "libopenblas64_"), Cdouble,
        (Ref{UInt8}, Ref{UInt8}, Ref{Int}, Ptr{Float64}, Ref{Int},
        Ptr{Float64}), 'F', A.uplo, n, A.data, lda, obj.work)
    vu = min(norminf, normfro)
    if vu < vl
        vu = 2*vl
    end
    abstol = 1e-8
    =#

    ccall((:dsyevr_64_, "libopenblas64_"), Cvoid,
    (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{Int},
        Ptr{Float64}, Ref{Int}, Ref{Float64}, Ref{Float64},
        Ref{Int}, Ref{Int}, Ref{Float64}, Ptr{Int},
        Ptr{Float64}, Ptr{Float64}, Ref{Int}, Ptr{Int},
        Ptr{Float64}, Ref{Int}, Ptr{Int}, Ref{Int},
        Ptr{Int}),
    obj.jobz, obj.range, A.uplo, n,
    A.data, lda, vl, vu,
    obj.il, obj.iu, abstol, obj.m,
    obj.w, obj.Z, obj.ldz, obj.isuppz,
    obj.work, obj.lwork, obj.iwork, obj.liwork,
    obj.info)

    k = obj.m[]
    λ, V = obj.w, obj.Z
    ldv = obj.ldz

    # V = V*diagm(sqrt.(λ))
    for j = 1:k
        tmp = sqrt(λ[j])
        for i = 1:n
            V[i,j] *= tmp
        end
    end

    # A = V*V'
    ccall((:dsyrk_64_, "libopenblas64_"), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{Int}, Ref{Int}, Ref{Float64},
            Ptr{Float64}, Ref{Int}, Ref{Float64}, Ptr{Float64}, Ref{Int}),
        A.uplo, 'N', n, k, 1.0, V, ldv, 0.0, A.data, lda)

    return A
end

