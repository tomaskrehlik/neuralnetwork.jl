function run()
    #load("optim.jl/src/init.jl")
    load("neuralnetworks.jl")
    #load("optim.jl/src/optimize.jl")
    #load("optim.jl/src/l_bfgs.jl")
    X = csvread("../checking_data/data.csv")
    y = csvread("../checking_data/output.csv")
    #Theta1 = csvread("Theta1.csv")
    #Theta2 = csvread("Theta2.csv")
    #params = [rollMatrix(Theta1), rollMatrix(Theta2)]

    input_layer_size  = 400
    hidden_layer_size = 25

    num_labels = 10
    lambda = 0
    data = zeros(length(y), 10)
    for i=1:10
        data[:,i] = ((i).==y)*1
    end

function f(x)
    E = nnCostFunction(x, input_layer_size, [hidden_layer_size], num_labels, X, data, lambda)[1]
end

function g(x)
    E = nnCostFunction(x, input_layer_size, [hidden_layer_size], num_labels, X, data, lambda)[2]
end
    results = optimize(f, g, randn(10285))
    @assert norm(results.minimum - [0.0, 0.0]) < 0.01
end
