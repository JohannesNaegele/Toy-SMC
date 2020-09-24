using Distributions
using Plots
using BenchmarkTools
using Optim
# using ForwardDiff
# using Traceur
using AdvancedMH
using MCMCChains
# using Turing
using Distributed


# cd("/home/johannes/Documents/GitHub/Toy-SMC/src/")
cd("C:/Users/Johannes/Nextcloud/Documents/GitHub/Toy-SMC/src")
# cd("C:/Users/econo/Nextcloud/Documents/GitHub/Toy-SMC/src")
include("simulate.jl") # include the data-generating function 

# addprocs(1)
# Threads.nthreads()

function likelihood(Y::Array{Float64,1}, x0, α, β, δ, σ, N = 1000)
    n = length(Y)
    P = zeros(n)
    v = TDist(2)
    w = Normal(0, σ^2) # different than in the paper (squared)
    u = Uniform() # needed for sampling
    X_normal = zeros(N)
    X_sample = zeros(N) # accepted
    b = zeros(N)
    c = convert(Array{Int}, zeros(N))
    V = zeros(N)
    Q = zeros(N)
    W = zeros(N) 
    for i in 1:n
        W[:] = rand(w, N)
        for j in 1:N
            if i == 1
                X_normal[j] = α + β * (x0 / (1 + x0^2)) + W[j]
            else
                X_normal[j] = α + β * (X_sample[j] / (1 + X_sample[j]^2)) + W[j]
            end
            V[j] = Y[i] - δ * X_normal[j]
            Q[j] = pdf(v, V[j])
        end
        # This evaluates the log-likelihood
        # I divide already here by N to ensure that P[i] doesn't get too big
        P[i] = sum(Q) / N
        # I generate quantiles for sampling
        Q .= Q ./ sum(Q)
        Q = cumsum(Q) 
        if true # i == 1
            # We resample with weights
            position = (rand(u, N) + [i for i in 0:(N-1)])/N
            l = 1
            k = 1
            while k < N
                if position[k] < Q[l]
                    X_sample[k] = X_normal[l]
                    k += 1
                else
                    l+=1
                end
            end            
            # x_sample wird automatisch in der nächsten Runde genutzt
        end
    end
    return sum(x->log(x), P)
end

n = 5000 # number of observations
N = 40000 # number of particles
α = 0.5; β = 0.3; δ = 1.; σ = 1.# parameters
# 0.729, 0.415, 0.538, 0.161 Ergebnis für likelihood
x0 = 1. # start value
Y = zeros(2, n)
simulate(x0, Y, α, β, δ, 1.) # generates hypothetical data
Y = Y[2,:] # only the observable data
# plot(Y)

@time likely = likelihood(Y, x0, 0.5, β, 1.0, 1., N)
@time likely = likelihood_tipp(Y, x0, 0.5, β, 1.0, 1., N)
@time likely = likelihood(Y, x0, 0.5, β, 1., 0.1, N)
@time likely = likelihood(Y, x0, 0.729, 0.415, 0.538, 0.161, N)
@time likely = likelihood_tipp(Y, x0, 0.729, 0.415, 0.538, 0.161, N)


approx(params) = likelihood(Y, x0, params[1], params[2], params[3], params[4], N)
opt(param) = -approx(param)
parameter = [0.4, 0.4, 1.1, 0.9]
@time optimum = optimize(opt, parameter, Optim.Options(iterations = 2000))
Optim.minimizer(optimum)

approx(Optim.minimizer(optimum))
approx([α, β, δ, σ])


approx(params) = likelihood(Y, x0, params, β, δ, σ, N)
opt(param) = -approx(param)
@time optimum = optimize(opt, -1, 2)
Optim.minimizer(optimum)

approx(Optim.minimizer(optimum))
approx(α)
