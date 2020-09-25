using Plots
using Distributions
using BenchmarkTools
using Optim
using AdvancedMH
using MCMCChains

cd("C:/Users/Johannes/Nextcloud/Documents/GitHub/Toy-SMC/src/Eigene_Funktion/")
include("simulate_eigen.jl") # include the data-generating function 

function likelihood(Y::Array{Float64,1}, x0, α, β, μ, τ, N = 10000)
    n = length(Y)
    P = zeros(n)
    v = TDist(τ)
    w = Normal(μ, 1)
    u = Uniform()
    # particles
    X_p = zeros(N)
    # sampled particles
    X_s = zeros(N)
    # normally distributed numbers
    W = zeros(N) 
    # t-distributed numbers
    V = zeros(N)
    # quantiles
    Q = zeros(N)
    for i in 1:n
        W[:] = rand(w, N)
        for j in 1:N
            if i == 1
                X_p[j] = x0
            else
                X_p[j] = X_s[j]*α + β/(1 + X_s[j]^2) + W[j]
            end
            V[j] = (Y[i] - X_p[j])*10
            Q[j] = pdf(v, V[j])
        end
        # This evaluates the likelihood for some time i; law of large numbers
        # I divide already here by N to ensure that P[i] doesn't get too big
        P[i] = sum(Q) / N
        # I generate quantiles for sampling
        Q .= Q ./ sum(Q)
        Q = cumsum(Q) 
        # We resample with weights
        position = (rand(u, N) + [i for i in 0:(N-1)])/N
        l = 1
        k = 1
        while k < N
            if position[k] < Q[l]
                X_s[k] = X_p[l]
                k += 1
            else
                l+=1
            end
        end            
        # X_s get's used in the next round
    end
    return sum(x->log(x), P)
end

n = 1000 # number of observations
N = 10000 # number of particles
x0 = 1.
α = 0.5; β = 2.0; μ = 1.; τ = 1. # parameters
Y = zeros(2, n)
simulate(Y, x0, α, β, τ, 1.) # generates hypothetical data
# Y = Y[2,:] # only the observable data

@time likely = likelihood(Y[2,:], x0, α, β, μ, τ, N)
# @time likely = likelihood(Y[2,:], x0, 0.55, β, μ, τ, N)

approx(params) = likelihood(Y[2,:], x0, params[1], params[2], params[3], params[4], N)
opt(param) = -approx(param)
parameter = [0.4, 2.2, 1.1, 0.9]
@time optimum = optimize(opt, parameter, Optim.Options(iterations = 100))
Optim.minimizer(optimum)

# julia> Optim.minimizer(optimum)
# 4-element Array{Float64,1}:
#  0.37118996237657903
#  2.541819049062401
#  1.2007515648766613
#  0.9753464747966533

approx([α, β, μ, τ])
approx(Optim.minimizer(optimum))

# julia> Optim.minimizer(optimum)
# 4-element Array{Float64,1}:
#  0.4789745373435752
#  1.4935466093527272
#  1.1710779161711162
#  0.9360542460215023
