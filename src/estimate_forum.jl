using Distributions
using Plots
using BenchmarkTools
# using Optim
# using ForwardDiff
# using Traceur
using AdvancedMH
using MCMCChains
# using Turing
using Distributed


# cd("/home/johannes/Documents/GitHub/Toy-SMC/src/")
cd("C:/Users/Johannes/Nextcloud/Documents/GitHub/Toy-SMC/src")
include("simulate.jl") # include the data-generating function 

addprocs(6)

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


n = 100 # number of observations
N = 2000 # number of particles
α = 0.5; β = 0.3; δ = 1.; σ = 1. # parameters
x0 = 1. # start value
Y = zeros(2, n)
simulate(x0, Y) # generates hypothetical data
Y = Y[1,:] # only the observable data
@time likely = likelihood(Y, x0, 0.5, β, 1.0, 1.1, N)

@btime likely = likelihood(Y, x0, α, β, δ, σ, N)
plot(Y)
α = 0.51; β = 0.3; δ = 1.; σ = 0.5 # parameters
@time likely = likelihood(Y, x0, α, β, δ, σ, N)
α = 10. # now it get's finally smaller
@btime likely = likelihood(Y, x0, α, β, δ, σ, N)

## additional: Metropolis-Hastings for α, β, δ, σ
prior = [Uniform(), Uniform(), Normal(0.7,2.0), Gamma(1.2,1.0)]
approx(params) = likelihood(Y, x0, params[1], params[2], params[3], params[4], N)
approx([α,β,δ,1.0])
model = DensityModel(approx)
# p1 = RWMH([Uniform(), Uniform(), Normal(1.2,2), Normal(0.18,1)])
# p1 = RWMH([Normal(),Normal(),Normal(),Normal()])
p1 = RWMH(prior)
@time chain = sample(model, p1, 200000; param_names=["α", "β", "δ", "σ"], chain_type=Chains)
println(chain)

prior = [Uniform(), Uniform(), Normal(0.7,2.0), Gamma(1.2,1.0)]
parameter = [0.1, 0.1, 0.7, 1.2]

function metropolis(prior, params, n::Int)
    chi = Uniform()
    proposal = MvNormal(4,0.05)
    # Step 0
    params_0 = params
    likeli_0 = approx(params_0)
    p_0 = prod(pdf.(prior, params_0))
    accept = zeros(100)
    ratio = 0.5
    c = 1.
    params_1 = 0
    for i in 1:n
        # Step 1
        params_1 = params_0 + c*rand(proposal)
        likeli_1 = approx(params_1) 
        p_1 = prod(pdf.(prior, params_1)) 
        q_0 = pdf(proposal, (params_1-params_0)/(c^2))
        q_1 = pdf(proposal, (params_0-params_1)/(c^2))
        if rand(chi) <= (exp(likeli_1-likeli_0)*p_1*q_1/(p_0*q_0))
            params_0 = params_1
            accept[((i-1) % 100 + 1)] = 1
        end 
        if i % 100 == 0 
            ratio = sum(accept)/100
            println(i)
            println(ratio)
            c = c/(ratio/0.234)
        end      
    end
    println(params_1)
end

metropolis(prior, parameter, 10000)