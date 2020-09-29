# x(t+1) = x*α + β/(1+x^2) + 0.1*w(t), where w(t) ~ N(μ, 1)
# y(t) = x(t) + 0.1*v(t), where v(t) ~ t(τ)

# using Plots
# using Distributions

function simulate(Y::Array{Float64,2}, x0 = 1.0, α = 0.5, β = 2.0, μ = 1.0, τ = 2.0)
    n = size(Y)[2]
    w = Normal(μ, 1)
    W = rand(w, n-1)
    v = TDist(τ)
    V = rand(v, n)
    Y[1,1] = x0
    Y[2,1] = Y[1,1] + 0.1*V[1]
    for i in 2:n
        Y[1,i] = Y[1,i-1]*α + β/(1+Y[1,i-1]^2) + 0.1*W[i-1]
        Y[2,i] = Y[1,i] + 0.1*V[i]
    end
end

# Y = zeros(2, 100)
# simulate(Y)
# plot(Y[1,:])
# plot(Y[2,:])
# Y[:,1]

