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


function likelihood(Y::Array{Float64,1}, P, V, Q, Q_last, W, X_normal, X_sample, x0, α, β, δ, σ, N=1000)
    n = length(Y)
    # P = zeros(n)
    u = Uniform()
    v = TDist(2)
    # X_normal = zeros(N)
    # X_sample = zeros(N)
    # V = zeros(N)
    # Q = zeros(N)
    # Q_last = ones(N)
    w = Normal(0, σ^2)
    # W = zeros(N) # rand(w, N)
    ρ = 0
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
            # P[i] += X_normal[j]
        end
        # See equation 8.11, FVA is just special case for ρ := 1
        if ρ == 0
            P[i] = log(sum(Q .* Q_last) / N)
        else
            P[i] = log(sum(Q) / N)
        end
        Q .= Q ./ sum(Q)
        ESS = N^2 / (sum(Q.^2))
        if ESS < N / 2
            ρ = 1
            println("hier")
        else
            ρ = 0
            println(ESS)
        end
        if ρ == 1            
            # in Quantiles for higher speed
            for j in 2:N
                Q[j] += Q[j - 1]
            end
            # if i==1
            #     println(Q)
            # end
            for j in 1:N
                step = rand(u)
                # k = 1
                # while Q[k] < step 
                #     k += 1
                # end
                K = 1
                L = Int(trunc(N / 2))
                M = N
                k = 1
                while K != M - 1 && k < N
                    # println(K,L,M)
                    if Q[K + L] < step
                        K = K + L
                        L = Int(trunc((M - K) / 2))
                    else
                        M = K + L
                        L = Int(trunc((M - K) / 2))
                        k += 1
                    end
                end  
                # println(L,K,M)
                if Q[K] < step
                    K += 1
                end    
                X_sample[j] = X_normal[K]
            end
        else
            Q_last .= Q
            X_sample .= X_normal
        end
    end
    return sum(P)
end

function main()
    n = 50
    N = 100000
    x0 = 1
    Y = zeros(2, n)
    simulate(x0, Y)
    Y = Y[1,:]
    P = zeros(n)
    X_normal = zeros(N)
    X_sample = zeros(N)
    V = zeros(N)
    Q = zeros(N)
    Q_last = ones(N)
    W = zeros(N)
    @time likely = likelihood(Y, P, V, Q, Q_last, W, X_normal, X_sample, x0, 0.5, 0.3, 1., 1., N)
    println(likely)
    approx(alpha) = likelihood(Y, x0, alpha[1], alpha[2], alpha[3], alpha[4], N)
    # optimize(approx, [0.55, 0.25], Optim.Options(iterations = 1000))

    # model = DensityModel(approx)
    # # p1 = RWMH([Normal(0.55, 2), Normal(0.25, 2), Normal(1.1, 2), Normal(0.9, 2)])
    # p2 = RWMH([Uniform(), Uniform(), Normal(1.3, 10), Normal(0.7, 10)])
    # println("n = 40, N = 100000, aber normalverteilte und schlechtere Priors")
    # p1 = RWMH([Normal(0.55, 2), Normal(0.25, 2), Normal(1.1, 2), Normal(0.9, 2)])
    # @time chain = sample(model, p1, 100000; param_names = ["α", "β", "δ", "σ"], chain_type = Chains)
    # # @time chain = sample(model, p2, 100000; param_names = ["α", "β", "δ", "σ"], chain_type = Chains)
    # println(chain)
end

main()