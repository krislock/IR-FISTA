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

