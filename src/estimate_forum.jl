using Distributions
# using Plots
# using BenchmarkTools
# using Optim
# using ForwardDiff
# using Traceur
# using AdvancedMH
# using MCMCChains
# using Turing


cd("/home/johannes/Documents/GitHub/Toy-SMC/src/")
include("simulate.jl") # include the data-generating function


function likelihood(Y::Array{Float64,1}, x0, α, β, δ, σ, N = 1000)
    n = length(Y)
    P = zeros(n)
    v = TDist(2)
    w = Normal(0, σ^2) # different than in the paper (squared)
    u = Uniform() # needed for sampling
    X_normal = zeros(N)
    X_sample = zeros(N)
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
        # multinomial sampling, I generate a uniform distributed value and divide the Q array in half and look in which part it is (and so on)
        for j in 1:N
            step = rand(u)
            K = 1
            L = Int(trunc(N / 2))
            M = N
            k = 1
            while K != M - 1 && k < N
                if Q[K + L] < step
                    K = K + L
                    L = Int(trunc((M - K) / 2))
                else
                    M = K + L
                    L = Int(trunc((M - K) / 2))
                    k += 1
                end
            end  
            if Q[K] < step
                K += 1
            end    
            X_sample[j] = X_normal[K] # save the sampled values 
        end
    end
    return sum(x->log(x), P)
end

n = 50 # number of observations
N = 100000 # number of particles
α = 0.5; β = 0.3; δ = 1.; σ = 1. # parameters
x0 = 1 # start value
Y = zeros(2, n)
simulate(x0, Y) # generates hypothetical data
Y = Y[1,:] # only the observable data
# plot(Y)
@time likely = likelihood(Y, x0, α, β, δ, σ, N)
α = 1. # no real difference
@time likely = likelihood(Y, x0, α, β, δ, σ, N)
α = 10. # now it get's finally smaller
@time likely = likelihood(Y, x0, α, β, δ, σ, N)

## additional: Metropolis-Hastings for α and β

# approx(params) = likelihood(Y, x0, params[1], params[2], 1., 1., N)
# model = DensityModel(approx)
# p1 = RWMH([Normal(0.55,2), Normal(0.25,2)])
# @time chain = sample(model, p1, 60000; param_names=["α", "β"], chain_type=Chains)