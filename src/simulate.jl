using Distributions
using Plots


function simulate(x0, Y::Array{Float64,2})
    n = size(Y)[2]
    w = Normal()
    W = rand(w, n)
    v = TDist(2)
    V = rand(v, n)
    Y[1,1] = 0.5 + 0.3*(x0/(1+x0^2)) + W[1]
    Y[2,1] = Y[1,1] + V[1]
    for i in 2:n
        Y[1,i] = 0.5 + 0.3*(Y[1,i-1]/(1+Y[1,i-1]^2)) + W[i]
        Y[2,i] = Y[1,i] + V[i]
    end
end

n = 20
x0 = 1
Y = zeros(2,n)
simulate(x0, Y)
Y = Y[1,:]
plot(Y)
# v = TDist(2)
# a = pdf(v, 1)