using Optim
using ForwardDiff
using Traceur
using Plots
using BenchmarkTools
using AdvancedMH
using Distributions
using MCMCChains
using Turing


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
    W = zeros(N) #rand(w, N)
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
        # in Quantiles for higher speed
        for j in 2:N
            Q[j] += Q[j-1]
        end
        # if i==1
        #     println(Q)
        # end
        # trick = Dict(Pair.(Q,X_normal))
        # sort(Q, by=x->trick[x])
        for j in 1:N
            step = rand(u)
            # k = 1
            # while Q[k] < step 
            #     k += 1
            # end
            K = 1
            L = Int(trunc(N/2))
            M = N
            k = 1
            while K != M-1 && k < 10000
                # println(K,L,M)
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
            # X_sample[j] = getindex.(Ref(trick),Q[k])
        end
    end
    return prod(P)
end

n = 50
N = 10000
x0 = 1
Y = zeros(2,n)
simulate(x0, Y)
Y = Y[1,:]
# plot(Y)
@time likely = likelihood(Y, x0, 0.5, 0.3, 1., 1., 10000)
P = zeros(n)
X_normal = zeros(N)
X_sample = zeros(N)
V = zeros(N)
Q = zeros(N)
W = zeros(N) #rand(w, N)
a = Dict(Pair.(Q,X_normal))
a[0.498743]
getindex.(Ref(a),[1])
@time likely = likelihood(Y, P, V, Q, W, X_normal, X_sample, x0, 0.55, 0.3, 1., 1., 10000)
@trace likely = likelihood(Y, x0, 0.55, 0.3, 1., 1., 5)
@code_warntype likelihood(Y, x0, 0.55, 0.3, 1., 1., 5)
# "Once I have solved
# the model, I use the Kalman filter to evaluate the likelihood of the model, given some
# parameter values. The whole process takes less than 1 second per evaluation of the
# likelihood."

approx(alpha) = -likelihood(Y, x0, alpha[1], alpha[2], 1., 1., 10000)
optimize(approx, [0.6, 0.25], Optim.Options(iterations = 500))

model = DensityModel(approx)
p1 = RWMH([Normal(0.55,1), Normal(0.25,1)])
chain = sample(model, p1, 100; param_names=["α", "β"], chain_type=Chains)