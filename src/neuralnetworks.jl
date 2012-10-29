function sigmoid(x, c)
  return 1/(1+exp(-c*x))
end

function sigmoidGradient(x, c)
  val = sigmoid(x, c)
  return val.*(1-val)
end

function feedForward(activation, theta)
  return activation*theta
end


#
# COST FUNCTION - the main function of the programme, outputs error function as well as gradient function.
#

function nnCostFunction(nnParameters, inputLayerSize::Integer, hiddenLayerSize::Array, outputSize::Integer, X::Array, y, lambda)


  #
  #  C H E C K I N G   T H E   I N P U T   P A R M E T E R S
  #
  
  # There are n+1 matrices, because the first parameter matrix goes from inputLayer
  numHiddenLayers = length(hiddenLayerSize)
  
  # Check that the input parameters to the function are fine
  params = (inputLayerSize + 1)*hiddenLayerSize[1]
  if numHiddenLayers > 1
    for i=1:(numHiddenLayers-1)
      params += (hiddenLayerSize[i]+1)*hiddenLayerSize[i+1]
    end
  end
  params += (hiddenLayerSize[numHiddenLayers]+1)*outputSize
  if length(nnParameters)!=params
    println("The inputed vector of parameters does not match the size implied by other parameters.")
    println(strcat("You input vector of length: ",length(nnParameters)))
    println(strcat("The right amount implied by neural network is: ",params))
    return false
  elseif min(size(y))!=outputSize
    println("The inputed data does not match the outputSize which has been inserterted!")
    return false
  end

  print("The parameters are of dimension: ")
  println(size(nnParameters))
  print("The data are of dimension: ")
  println(size(X))
  print("The output is of dimension: ")
  println(size(y))
  print("\n\n\n")
  #
  # S E T T I N G   U P   U S E F U L   V A R I A B L E S
  #

  # A dummy variable to keep track of number of parameters I already reshaped
  wentThroughParameters = 0
  regularization = 0
  # Connecting the layers so I can do it in one loop
  m = size(X)[1]
  layerSizes = [inputLayerSize hiddenLayerSize outputSize]
  numLayers = length(layerSizes)

  activation = [ones(m) X] # First input is the dataset

  zs = Array(Float64, m*(sum(layerSizes[1:numLayers]) + numLayers - 1 )) # The unrolled backpropagation
  zsUnrolled = 0 # Number of the parameters I have already went through, didnt come up with better idea how to do this...
  zs[1:length(activation)] = rollMatrix(activation)
  zsUnrolled = length(activation)
  #
  # F O R W A R D   P R O P A G A T I O N
  #
  for i=1:(numLayers - 1)
    (activation, regularization, wentThroughParameters, zs, zsUnrolled) = propagate(wentThroughParameters, layerSizes[i], layerSizes[i+1], nnParameters, activation, regularization, m, zs, zsUnrolled)
  end

  output = activation
  # There are ones as a byproduct of the propagation, need to remove them
  output = output[:,2:size(output)[2]]


  # Computing the cost
  delta_output = output - y
  E = 0

  tr = log(output)
  fa = log(1-output)

  E = - 1/m * sum(y.*tr + abs(y-1).*fa) + regularization
  #return E
  #return E = 1/2 * sum(norm(delta_output)^2) + lambda/2 * regularization

  #
  # B A C K P R O P A G A T I O N
  #

  # 1) compute the error
  # done in delta_output
  # 2) write for loop to backpropagate
  delta = delta_output
  activation_previous = rollMatrix(output)
  Delta = Array(Float64)
  for i=1:(numLayers-1)
    #print("Step: ")
    #print(i)
    #print("\n")
    #print("Size of delta matrix: ")
    #print(size(delta))
    #print("\n")
    print("Step number ")
    println(i)
    (Delta, delta, activation_previous) = backpropagate(delta, nnParameters, layerSizes, i, m, zs, activation_previous, Delta, lambda)
    delta = delta[:,2:size(delta)[2]]
  end
  return E, Delta
end


function backpropagate(delta_previous, nnParameters, layerSizes, step, m, zs, activation_previous, Delta, lambda)
  numLayers = length(layerSizes)
  fromTheta = sum(((layerSizes[1:(numLayers-step-1)]+1).*(layerSizes[2:(numLayers-step)])))+1
  stepprev = step - 1
  fromThetaPrev = sum(((layerSizes[1:(numLayers-stepprev-1)]+1).*(layerSizes[2:(numLayers-stepprev)])))
  leftTheta = layerSizes[numLayers - step] + 1
  rightTheta = layerSizes[numLayers - step + 1]
  toTheta = fromTheta + (leftTheta)*rightTheta - 1
  Theta = unRollMatrix(nnParameters[fromTheta:toTheta], rightTheta, leftTheta)
  # = (number of parameters + number of constants) - number of parameters up to this step
  fromZ = (sum(layerSizes[1:(numLayers-step-1)]) + (numLayers - step-1) )
  if numLayers-step-1==0 then
      fromZ = 0
  end
  toZ = (fromZ + layerSizes[numLayers - step]) * m
  fromZ = fromZ * m + 1
  z = unRollMatrix(zs[fromZ:toZ], m, (layerSizes[numLayers - step]))
  gr = [ones(m) sigmoidGradient(z,1)[:,1:size(z)[2]]]
  activation = [ones(m) sigmoid(z,1)[:,1:(size(z)[2])]]
  z = [ones(m) z[:,2:size(z)[2]]]
  delta = (delta_previous*Theta).*(gr)
  Delta_new = 1/m * delta_previous'*activation + lambda/m * [zeros(size(Theta)[2]) Theta[2:(size(Theta)[1]),:]' ]'
  Delta = [Delta, rollMatrix(Delta_new)]
  return (Delta, delta, activation)
end


function propagate(wentThroughParameters::Integer, left::Integer, right::Integer, nnParameters, input, regularization, m, zs, zsUnrolled::Integer)


  # Reconstruct Theta
  c = ones(m)
  from = wentThroughParameters + 1
  to = wentThroughParameters + (left + 1)*right
  fromZ = zsUnrolled + 1
  toZ = (right)*m + (fromZ - 1)

  wentThroughParameters += to-from + 1
  Theta = unRollMatrix(nnParameters[from:to], right, (left + 1))

  # Input already includes ones on the left side
  # Premultiply by theta to get Z
  z = input*Theta'
  # Do sigmoid to get the activation
  activation = sigmoid(z, 1)

  # Add ones for the next itteration
  output = [c activation]
  zs[fromZ:toZ] = rollMatrix(z)
  regularization += 1/(2*m) * sum(Theta[:,2:(size(Theta)[2])].^2)

  return (output, regularization, wentThroughParameters, zs, toZ)
end

function rollMatrix(M)
  return reshape(M, length(M), 1)
end

function unRollMatrix(vec, rows, cols)
  return reshape(vec, rows, cols)
end
