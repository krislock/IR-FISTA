function isdiagallpos(X)
    return all(@inbounds X[i,i] > 0 for i=1:size(X,1))
end

function plusdiag!(M::AbstractArray{T,2}, y::Vector{T}) where T
    for i=eachindex(y)
        @inbounds M[i,i] += y[i]
    end
    return M
end

# Replaces linesearch function evaluations with last function evaluation for cleaner plots
function cleanvals!(vals, linesearchcalls)
    fgcalls = sum(linesearchcalls)
    v = view(vals, length(vals)-fgcalls+1:length(vals))
    a = 1
    for numcalls in linesearchcalls
        b = a + numcalls - 1
        fill!(view(v, a:b), v[b])
        a = b + 1
    end
    return vals
end