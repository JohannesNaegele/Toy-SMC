using Plots
using BenchmarkTools
using Distributions
# using Optim
# using ForwardDiff
# using Traceur
# using AdvancedMH
# using MCMCChains
# using Turing


cd("/home/johannes/Documents/GitHub/Toy-SMC/src/")
include("simulate.jl")


function likelihood(Y::Array{Float64,1}, x0, α, β, δ, σ, N=1000)
    n = length(Y)
    P = zeros(n)
    u = Uniform()
    v = TDist(2)
    X_normal = zeros(N)
    X_sample = zeros(N)
    V = zeros(N)
    Q = zeros(N)
    w = Normal(0, σ^2)
    W = zeros(N) 
    for i in 1:n
        W[:] = rand(w, N)
        for j in 1:N
            if i == 1
                X_normal[j] = α + β*(x0/(1+x0^2)) + W[j]
            else
                X_normal[j] = α + β*(X_sample[j]/(1+X_sample[j]^2)) + W[j]
            end
            V[j] = Y[i] - δ*X_normal[j]
            Q[j] = pdf(v, V[j])
            # P[i] += X_normal[j]
        end
        # I divide already here to ensure that P[i] doesn't get too big
        P[i] = sum(Q)/N
        Q .= Q ./ sum(Q)

        for j in 1:N
            step = rand(u)
            K = 1
            L = Int(trunc(N/2))
            M = N
            k = 1
            while K != M-1 && k < N
                if Q[K+L] < step
                    K = K+L
                    L = Int(trunc((M-K)/2))
                else
                    M = K+L
                    L = Int(trunc((M-K)/2))
                    k += 1
                end
            end  
            # println(L,K,M)
            if Q[K] < step
                K += 1
            end    
            X_sample[j] = X_normal[K]
        end
    end
    return sum(x -> log(x), P)
end

n = 50 # number of observations
N = 100000 # number of particles
x0 = 1 # start value
Y = zeros(2,n)
simulate(x0, Y) # generates hypothetical data
Y = Y[1,:] # only the observable data
# plot(Y)
@btime likely = likelihood(Y, x0, 0.5, 0.3, 1., 1., N)
likely = likelihood(Y, x0, 1., 0.3, 1., 1., N)

# approx(alpha) = likelihood(Y, x0, alpha[1], alpha[2], 1., 1., 40000)
# model = DensityModel(approx)
# p1 = RWMH([Normal(0.55,2), Normal(0.25,2)])
# @time chain = sample(model, p1, 60000; param_names=["α", "β"], chain_type=Chains)