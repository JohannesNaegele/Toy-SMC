using Distributions

function simulate(x0, Y::Array{Float64,2})
    n = size(Y)[2]
    u = Normal()
    U = rand(u, n-1)
    v = TDist(2)
    V = rand(v, n)
    Y[1,1] = x0
    Y[2,1] = Y[1,1] + V[1]
    for i in 2:n
        Y[1,i] = 0.5 + 0.3*(Y[1,i-1]/(1+Y[1,i-1]^2)) + U[i-1]
        Y[2,i] = Y[1,i] + V[i]
    end
end

n = 20
x0 = 1
Y = zeros(2,n)
simulate(x0, Y)
println(i)