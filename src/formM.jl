#=
[
H.*H.*()  -Diag()    -Id()
-diag()      0           0
Λ*()         0        ()*X
]
=#

Rd = H.*H.*(X - G) - Diagonal(y) - Λ
rp = 1 .- diag(X)
Rc = Λ*X
@show norm(Rd), norm(rp), norm(Rc)
display([eigvals(X) eigvals(Λ)])

nx = n^2
ny = n
nλ = n^2

nv = nx + ny + nλ

xinds = 1:nx
yinds = nx+1:nx+ny
λinds = nx+ny+1:nv

v = zeros(nv)

M = zeros(nv, nv)
for j in eachindex(v)
    v[j] = 1.0

    vx = view(v, xinds)
    vy = view(v, yinds)
    vλ = view(v, λinds)

    vX = reshape(vx, n, n)
    vΛ = reshape(vλ, n, n)

    R1 = H.*H.*vX - Diagonal(vy) - vΛ
    r2 = -diag(vX)
    R3 = Λ*vX + vΛ*X

    M[:,j] = [R1[:]; r2; R3[:]]

    v[j] = 0.0
end

b = [-Rd[:]; -rp; -Rc[:]]

dv = M\b

dx = view(dv, xinds)
dy = view(dv, yinds)
dλ = view(dv, λinds)

dX = Symmetric(reshape(dx, n, n))
dΛ = Symmetric(reshape(dλ, n, n))

X = X + dX
y = y + dy
Λ = Λ + dΛ

Rd = H.*H.*(X - G) - Diagonal(y) - Λ
rp = 1 .- diag(X)
Rc = Λ*X
@show norm(Rd), norm(rp), norm(Rc)
display([eigvals(X) eigvals(Λ)])

