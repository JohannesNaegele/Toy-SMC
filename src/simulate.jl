function simulate(x0, Y::Array{Float64,2}, α = 0.5, β = 0.3, δ = 1., σ = 1.)
    n = size(Y)[2]
    w = Normal(0, σ^2)
    W = rand(w, n)
    v = TDist(2)
    V = rand(v, n)
    Y[1,1] = α + β * (x0 / (1 + x0^2)) + W[1]
    Y[2,1] = δ * Y[1,1] + V[1]
    for i in 2:n
        Y[1,i] = α + β * (Y[1,i - 1] / (1 + Y[1,i - 1]^2)) + W[i]
        Y[2,i] = δ * Y[1,i] + V[i]
    end
end