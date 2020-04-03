cd("/home/johannes/Documents/GitHub/Toy-SMC/src/")
include("simulate.jl")

function likelihood(Y::Array{Float64,1}, x0, N=1000)
    n = size(Y)[2]
    u = Normal()
    X_normal = zeros(N)
    X_sample = zeros(N)
    V = zeros(N)
    Q = zeros(N)
    w = Normal()
    W = zeros(N) #rand(w, N)
    for i in 1:n
        W[:] = rand(w, N)
        for j in 1:N
            if i=1
                X_normal[j] = 0.5 + 0.3*(x0/(1+x0^2)) + W[j]
            else
                X_normal[j] = 0.5 + 0.3*(X_sample[j]/(1+X_sample[j]^2)) + W[j]
            end
            V[j] = Y[j] - X_normal[j]
            Q[j] = pdf(v, V[j])
        end
        Q .= Q ./ sum(Q)
        # in Quantilesfor higher speed
        for j in 2:N
            Q[j] += Q[j-1]
        end
        for j in 1:N
            step = rand(u)
            k = 1
            while Q[k] < step 
                k++
            end
            X_sample[j] = X_normal[k]
        end
    end
end