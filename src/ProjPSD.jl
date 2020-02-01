using LinearAlgebra

function dsyevd!(jobz, uplo, n, A, lda,
        w, work, lwork, iwork, liwork, info)
    ccall((:dsyevd_64_, "libopenblas64_"), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{Int}, Ref{Float64}, Ref{Int},
         Ref{Float64}, Ref{Float64}, Ref{Int}, Ref{Int}, Ref{Int},
         Ref{Int}),
        jobz, uplo, n, A, lda,
        w, work, lwork, iwork, liwork, info)
end

function dsyrk!(uplo, trans, n, k,
        alpha, A, lda, beta, C, ldc)
    ccall((:dsyrk_64_, "libopenblas64_"), Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{Int}, Ref{Int},
            Ref{Float64}, Ref{Float64}, Ref{Int},
            Ref{Float64}, Ref{Float64}, Ref{Int}),
        uplo, trans, n, k,
        alpha, A, lda, beta, C, ldc)
end

struct ProjPSD
    nmax::Int
    jobz::Char
    w::Vector{Float64}
    work::Vector{Float64}
    lwork::Int
    iwork::Vector{Int}
    liwork::Int
    info::Base.RefValue{Int}

    function ProjPSD(nmax)
        n = nmax
        A = Symmetric(zeros(1,1))

        jobz = 'V'
        lda = n
        w = zeros(n)
        info   = Ref{Int}(0)

        # Perform an optimal workspace query
        lwork = -1
        liwork = -1
        work   = zeros(1)
        iwork  = zeros(Int, 1)
        dsyevd!(jobz, A.uplo, n, A.data, lda, w,
                work, lwork, iwork, liwork, info)
        lwork = Int(real(work[1]))
        liwork = iwork[1]
        resize!(work, lwork)
        resize!(iwork, liwork)

        new(nmax, jobz, w, work, lwork, iwork, liwork, info)
    end
end


function (obj::ProjPSD)(A::Symmetric, negAminus::Symmetric)
    lda, n = size(A)

    @assert lda == n
    @assert n <= obj.nmax

    dsyevd!(obj.jobz, A.uplo, n, A.data, lda, obj.w,
            obj.work, obj.lwork, obj.iwork, obj.liwork, obj.info)

    λ, V = obj.w, A.data

    # V = V*Diagonal(sqrt.(abs.(λ)))
    k = 0
    @inbounds for j = 1:n
        tmp = λ[j]
        if tmp < 0.0
            k = j
            tmp = -tmp
        end
        tmp = sqrt(tmp)
        for i = 1:n
            V[i,j] *= tmp
        end
    end

    # A = V*V'
    # A := alpha*V*V**T + beta*A, alpha = 1.0, beta = 0.0
    if k > 0
        dsyrk!(negAminus.uplo, 'N', n, k, 1.0, V, n, 0.0,
               negAminus.data, n)
    else
        negAminus.data .= 0.0
    end

    if k < n
        dsyrk!(A.uplo, 'N', n, n-k, 1.0, view(V,:,k+1:n), n, 0.0,
               A.data, n)
    else
        A.data .= 0.0
    end

    return A
end

############################################################

