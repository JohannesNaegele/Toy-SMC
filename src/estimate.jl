using Optim

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
        # I divide already here to ensure that P[i] doesn't get to big
        P[i] = sum(Q)/N
        Q .= Q ./ sum(Q)
        # in Quantiles for higher speed
        for j in 2:N
            Q[j] += Q[j-1]
        end
        # if i==1
        #     println(Q)
        # end
        for j in 1:N
            step = rand(u)
            k = 1
            while Q[k] < step 
                k += 1
            end
            X_sample[j] = X_normal[k]
        end
    end
    return prod(P)
end

n = 100
x0 = 1
Y = zeros(2,n)
simulate(x0, Y)
Y = Y[1,:]
# plot(Y)
likely = likelihood(Y, x0, 0.55, 0.3, 1., 1., 5000)

approx(alpha) = -likelihood(Y, x0, alpha[1], alpha[2], 1., 1., 5000)*1e70
optimize(approx, [0.6, 0.3], Optim.Options(iterations = 100))
