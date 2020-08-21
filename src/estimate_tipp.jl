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
# cd("C:/Users/Johannes/Nextcloud/Documents/GitHub/Toy-SMC/src")
cd("C:/Users/econo/Nextcloud/Documents/GitHub/Toy-SMC/src")
include("simulate.jl") # include the data-generating function 

# addprocs(1)

Threads.nthreads()

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
        Threads.@threads for j in 1:N
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
        Threads.@threads for j in 1:N
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

function likelihood_tipp(Y::Array{Float64,1}, x0, α, β, δ, σ, N = 1000)
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
        Threads.@threads for j in 1:N
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
        Q = cumsum(Q) * N
        if true # i == 1
            # We resample with weights
            v1 = rand(u)*N
            b[1] = trunc(Q[1]-v1)
            b0 = trunc(0.0-v1)
            c[1] = Int(b[1] - b0)
            # println(c[1])
            X_sample[1:c[1]] .= X_normal[1]
            j = c[1]
            for sim in 2:N 
                b[1] = Int(trunc(Q[1]-v1))
                b0 = Int(trunc(0.0-v1))
                c[1] = Int(b[sim] - b[sim - 1])
                X_sample[j+1:j+c[sim]] .= X_normal[sim]
                j += c[sim]
            end
            # x_sample wird automatisch in der nächsten Runde genutzt
        end
    end
    return sum(x->log(x), P)
end


n = 1000 # number of observations
N = 10000 # number of particles
α = 0.5; β = 0.3; δ = 1.; σ = 1.# parameters
# 0.729, 0.415, 0.538, 0.161 Ergebnis
x0 = 1. # start value
Y = zeros(2, n)
simulate(x0, Y, α, β, δ, 1.) # generates hypothetical data
Y = Y[2,:] # only the observable data
# plot(Y)
@time likely = likelihood(Y, x0, 0.5, β, 1.0, 1., N)
@time likely = likelihood(Y, x0, 0.5, β, 1., 0.1, N)
@time likely = likelihood_tipp(Y, x0, α, β, δ, 0.8, N)
@time likely = likelihood_tipp(Y, x0, α, β, δ, 1., N)
