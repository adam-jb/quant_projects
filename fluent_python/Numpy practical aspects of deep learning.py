# @@ Cell 1
# normalising inputs speeds up training process and gets better results, as the search space becomes 
# more symmetric (means that a single learning rate is appropriate for all dimensions with doing gradient descent)
# If data are on roughly the same scale it won't hurt to normalise inputs, but will make less of a difference


# weights increase/decrease by whatever they are to the power of the number of layers they are from the 
# output layer. So if lots of layers separate them the gradient can easily go to a very small (~0) or very 
# big number. This is the vanishing/exploding gradient. More on this:
# https://www.coursera.org/learn/deep-neural-network/lecture/C9iQO/vanishing-exploding-gradients


# something that helps exploding/vanishing network: careful weights initialisation
# the more weights you have the smaller you want individual weights to be
# to solve this, it's common to initialise random weights with:
however_many_weights = 6
n_L_minusOne = 5   # no. of nodes in preceding layer
weights_init_one_layer = np.random.randn(however_many_weights) * np.sqrt(2 / n_L_minusOne) # is 1, not 2, if not relu
weights_init_one_layer

# @@ Cell 2
## Notes on Practical Aspects of Deep Learning (week 1 of course 2 in deep learning specialisation)

## making a deep NN in numpy with added:
# > regularisation (drop out)
# > input normalisation
# > improved weight initialisation



## Aiming to predict whether a plant is species setosa or not

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# @@ Cell 3
### Activation functions:
# denote generic activation function as g(x)

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1- sigmoid_output)   # * = element-wise mult; @ = matrix mult


# ReLu: better for larger values as the gradient doesn't become close to 0, which it does for tanh and sigmoid
# ReLu trains faster because of this 
def relu(x):
    return np.where(x >= 0, x, 0)

def relu_derivative(x):       
    """note x is the original input to relu, not the output from relu, unlike other
    derivatives of activation functions"""
    return np.where(x >= 0, 1, 0)

# @@ Cell 4
# normalising inputs speeds up training process and gets better results, as the search space becomes 
# more symmetric (means that a single learning rate is appropriate for all dimensions with doing gradient descent)
# If data are on roughly the same scale it won't hurt to normalise inputs, but will make less of a difference


# weights increase/decrease by whatever they are to the power of the number of layers they are from the 
# output layer. So if lots of layers separate them the gradient can easily go to a very small (~0) or very 
# big number. This is the vanishing/exploding gradient. More on this:
# https://www.coursera.org/learn/deep-neural-network/lecture/C9iQO/vanishing-exploding-gradients


# something that helps exploding/vanishing network: careful weights initialisation
# the more weights you have the smaller you want individual weights to be
# to solve this, it's common to initialise random weights with:
however_many_weights = 6
n_L_minusOne = 5   # no. of nodes in preceding layer
weights_init_one_layer = np.random.randn(however_many_weights) * np.sqrt(2 / n_L_minusOne) # is 1, not 2, if not relu
weights_init_one_layer

# @@ Cell 5
## Notes on Practical Aspects of Deep Learning (week 1 of course 2 in deep learning specialisation)

## making a deep NN in numpy with added:
# > regularisation (drop out)
# > input normalisation
# > improved weight initialisation



## Aiming to predict whether a plant is species setosa or not

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# @@ Cell 6
## some regularisation components you can add
# lambda becomes another hyperparameter to tune in your model


def l2_regularisation(lambda_val, m, w):
    """aka 'L2 norm' or 'Euclidean norm' or 'weight decay' (as causes weights to get smaller)
    
    Much more popular than L1 regularisation
    """
    return lambda_val * np.power(np.sum(w), 2) / (m * 2)

## (if using L-2 regularisation update cost function too)




### this bit by itself is called the Frobenius Norm:
# np.power(np.sum(w), 2)


def l1_regularisation(lambda_val, m, w):
    """ aka L1 norm"""
    return (lambda_val * np.sum(np.absolute(w))) / m




## to include l2_regularisation during backpropagation we add a term:
# dW[L] = (what it would be normally) + (lambda / m)*W[L]   
## where new term is the derivative of l2_regularisation func with respect to W

# @@ Cell 7
# normalising inputs speeds up training process and gets better results, as the search space becomes 
# more symmetric (means that a single learning rate is appropriate for all dimensions with doing gradient descent)
# If data are on roughly the same scale it won't hurt to normalise inputs, but will make less of a difference


# weights increase/decrease by whatever they are to the power of the number of layers they are from the 
# output layer. So if lots of layers separate them the gradient can easily go to a very small (~0) or very 
# big number. This is the vanishing/exploding gradient. More on this:
# https://www.coursera.org/learn/deep-neural-network/lecture/C9iQO/vanishing-exploding-gradients


# something that helps exploding/vanishing network: careful weights initialisation
# the more weights you have the smaller you want individual weights to be
# to solve this, it's common to initialise random weights with:
however_many_weights = 6
n_L_minusOne = 5   # no. of nodes in preceding layer
weights_init_one_layer = np.random.randn(however_many_weights) * np.sqrt(2 / n_L_minusOne) 
                    # best to set numerator in the above to 1, not 2, if not relu
weights_init_one_layer


# there are other initialisation formulae in the vein of the above including:
# Xavier initialisation (involves using tanh())

# deciding on multiplier to use in weights initialisation (instead of 2 in the above)
# could become another hyperparameter to tune

# more on initialising weights:
# https://www.coursera.org/learn/deep-neural-network/lecture/RwqYe/weight-initialization-for-deep-networks

# @@ Cell 8
### Activation functions:
# denote generic activation function as g(x)

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1- sigmoid_output)   # * = element-wise mult; @ = matrix mult


# ReLu: better for larger values as the gradient doesn't become close to 0, which it does for tanh and sigmoid
# ReLu trains faster because of this 
def relu(x):
    return np.where(x >= 0, x, 0)

def relu_derivative(x):       
    """note x is the original input to relu, not the output from relu, unlike other
    derivatives of activation functions"""
    return np.where(x >= 0, 1, 0)

# @@ Cell 9
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


print(*data.shape)  # return values outside tuple
print(data.shape)

# @@ Cell 10
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise data 




print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

# @@ Cell 11
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
print(np.linalg.norm(data, axis = 1))




print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

# @@ Cell 12
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
print(np.linalg.norm(data[:, :4], axis = 1))




print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

# @@ Cell 13
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
print(np.linalg.norm(data[:, :4]))




print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

# @@ Cell 14
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
print(np.linalg.norm(data[:, :4], axis = 0))




print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

# @@ Cell 15
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
norms_each_column = np.linalg.norm(data[:, :4], axis = 0))
print(norms_each_column)



print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

# @@ Cell 16
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
norms_each_column = np.linalg.norm(data[:, :4], axis = 0)
print(norms_each_column)



print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

# @@ Cell 17
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
data = data.values.astype(float)
norms_each_column = np.linalg.norm(data[:, :4], axis = 0)
print(norms_each_column)



print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

# @@ Cell 18
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
data = data.values.astype(float)
norms_each_column = np.linalg.norm(data.astype(np.float([:, :4], axis = 0)
print(norms_each_column)



print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

# @@ Cell 19
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
data = data.values.astype(float)
norms_each_column = np.linalg.norm(data.astype(np.float())[:, :4], axis = 0)
print(norms_each_column)



print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

# @@ Cell 20
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
norms_each_column = np.linalg.norm(data.astype(np.float())[:, :4], axis = 0)
print(norms_each_column)



print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

# @@ Cell 21
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
norms_each_column = np.linalg.norm(data)[:, :1], axis = 0)
print(norms_each_column)



print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

# @@ Cell 22
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
norms_each_column = np.linalg.norm(data)[:, :1], axis = 0
print(norms_each_column)



print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

# @@ Cell 23
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
norms_each_column = np.linalg.norm(data[:, :1], axis = 0)
print(norms_each_column)



print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

# @@ Cell 24
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
norms_each_column = np.linalg.norm(data[:, 1], axis = 0)
print(norms_each_column)



print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

# @@ Cell 25
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, 1], axis = 0)
    data[:, 1] = data[:, 1] / norms_each_column


print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

# @@ Cell 26
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, 1], axis = 0)
    data[:, 1] = data[:, 1] / norms_each_column

print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

plt.hist(data[:, 1])

# @@ Cell 27
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, 1], axis = 0)
    data[:, 1] = data[:, 1] / norms_each_column

print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

plt.hist(data[:, 1])
plt.hist(data[:, 2])

# @@ Cell 28
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, 1], axis = 0)
    data[:, 1] = data[:, 1] / norms_each_column
    print(norms_each_column)

print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

plt.hist(data[:, 1])
plt.hist(data[:, 2])

# @@ Cell 29
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, 1], axis = 0)
    data[:, 1] = data[:, 1] / norms_each_column
    print(norms_each_column)

    
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, 1], axis = 0)
    print(norms_each_column)

    
    
print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

plt.hist(data[:, 1])
plt.hist(data[:, 2])

# @@ Cell 30
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, 1], axis = 0)
    data[:, 1] = data[:, 1] / norms_each_column
    print(norms_each_column)

    
# np.linalg.norm calculates L2 norm, aka the sum of the squared values
for i in range(4): 
    print(np.min(data[:, 1]))
    print(np.max(data[:, 1]))
    print(np.mean(data[:, 1]))
    print(np.std(data[:, 1]))

    
print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

plt.hist(data[:, 1])
plt.hist(data[:, 2])

# @@ Cell 31
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, i], axis = 0)
    data[:, i] = data[:, i] / norms_each_column
    print(norms_each_column)

    
# np.linalg.norm calculates L2 norm, aka the sum of the squared values
for i in range(4): 
    print(np.min(data[:, i]))
    print(np.max(data[:, i]))
    print(np.mean(data[:, i]))
    print(np.std(data[:, i]))

    
print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

plt.hist(data[:, 1])
plt.hist(data[:, 2])

# @@ Cell 32
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, i], axis = 0)
    data[:, i] = data[:, i] / norms_each_column
    print(norms_each_column)

    
# np.linalg.norm calculates L2 norm, aka the sum of the squared values
for i in range(4): 
    print(np.min(data[:, i]))
    print(np.max(data[:, i]))
    print(np.mean(data[:, i]))
    print(np.std(data[:, i]))

    
print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

plt.hist(data[:, 1], alpha = 0.1)
plt.hist(data[:, 2], alpha = 0.1)
plt.hist(data[:, 3], alpha = 0.1)

# @@ Cell 33
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, i], axis = 0)
    data[:, i] = data[:, i] / norms_each_column
    print(norms_each_column)

    
# np.linalg.norm calculates L2 norm, aka the sum of the squared values
for i in range(4): 
    print(np.min(data[:, i]))
    print(np.max(data[:, i]))
    print(np.mean(data[:, i]))
    print(np.std(data[:, i]))

    
print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

plt.hist(data[:, 1], alpha = 0.2)
plt.hist(data[:, 2], alpha = 0.2)
plt.hist(data[:, 3], alpha = 0.2)

# @@ Cell 34
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, i], axis = 0)
    data[:, i] = data[:, i] / norms_each_column
    print(norms_each_column)

    
# np.linalg.norm calculates L2 norm, aka the sum of the squared values
"""
for i in range(4): 
    print(np.min(data[:, i]))
    print(np.max(data[:, i]))
    print(np.mean(data[:, i]))
    print(np.std(data[:, i]))
"""
    
print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

plt.hist(data[:, 1], alpha = 0.2)
plt.hist(data[:, 2], alpha = 0.2)
plt.hist(data[:, 3], alpha = 0.2)

# @@ Cell 35
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, i], axis = 0)
    data[:, i] = data[:, i] / norms_each_column

    
# np.linalg.norm calculates L2 norm, aka the sum of the squared values
"""
for i in range(4): 
    print(np.min(data[:, i]))
    print(np.max(data[:, i]))
    print(np.mean(data[:, i]))
    print(np.std(data[:, i]))
"""
    
print(*data.shape)  # return values outside tuple
print(data.shape)
print(data[:5, :])

plt.hist(data[:, 1], alpha = 0.2)
plt.hist(data[:, 2], alpha = 0.2)
plt.hist(data[:, 3], alpha = 0.2)

# @@ Cell 36
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, i], axis = 0)
    data[:, i] = data[:, i] / norms_each_column

    
# np.linalg.norm calculates L2 norm, aka the sum of the squared values
"""
for i in range(4): 
    print(np.min(data[:, i]))
    print(np.max(data[:, i]))
    print(np.mean(data[:, i]))
    print(np.std(data[:, i]))
"""
    
print(*data.shape)  # return values outside tuple
print(data.shape)

plt.hist(data[:, 1], alpha = 0.2)
plt.hist(data[:, 2], alpha = 0.2)
plt.hist(data[:, 3], alpha = 0.2)

# @@ Cell 37
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, i], axis = 0)
    data[:, i] = data[:, i] / norms_each_column

    
# np.linalg.norm calculates L2 norm, aka the sum of the squared values
"""
for i in range(4): 
    print(np.min(data[:, i]))
    print(np.max(data[:, i]))
    print(np.mean(data[:, i]))
    print(np.std(data[:, i]))
"""
    
print(*data.shape)  # return values outside tuple
print(data.shape)

plt.hist(data[:, 1], alpha = 0.2)
plt.hist(data[:, 2], alpha = 0.2)
plt.hist(data[:, 3], alpha = 0.2)
plt.show()

# @@ Cell 38
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, i], axis = 0)
    data[:, i] = data[:, i] / norms_each_column

    
# np.linalg.norm calculates L2 norm, aka the sum of the squared values
"""
for i in range(4): 
    print(np.min(data[:, i]))
    print(np.max(data[:, i]))
    print(np.mean(data[:, i]))
    print(np.std(data[:, i]))
"""
    
print(*data.shape)  # return values outside tuple
print(data.shape)


# distribution of values for each column shows they are operating on the same scale
plt.hist(data[:, 1], alpha = 0.2)
plt.hist(data[:, 2], alpha = 0.2)
plt.hist(data[:, 3], alpha = 0.2)
plt.hist(data[:, 4], alpha = 0.2)
plt.show()

# @@ Cell 39
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, i], axis = 0)
    data[:, i] = data[:, i] / norms_each_column

    
# np.linalg.norm calculates L2 norm, aka the sum of the squared values
"""
for i in range(4): 
    print(np.min(data[:, i]))
    print(np.max(data[:, i]))
    print(np.mean(data[:, i]))
    print(np.std(data[:, i]))
"""
    
print(*data.shape)  # return values outside tuple
print(data.shape)


# distribution of values for each column shows they are operating on the same scale
plt.hist(data[:, 0], alpha = 0.2)
plt.hist(data[:, 1], alpha = 0.2)
plt.hist(data[:, 2], alpha = 0.2)
plt.hist(data[:, 3], alpha = 0.2)
plt.show()

# @@ Cell 40
## Notes on Practical Aspects of Deep Learning (week 1 of course 2 in deep learning specialisation)

## making a deep NN in numpy with added:
# > regularisation (drop out)
# > input normalisation
# > improved weight initialisation



## Aiming to predict whether a plant is species setosa or not

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# @@ Cell 41
## some regularisation components you can add
# lambda becomes another hyperparameter to tune in your model


def l2_regularisation(lambda_val, m, w):
    """aka 'L2 norm' or 'Euclidean norm' or 'weight decay' (as causes weights to get smaller)
    
    Much more popular than L1 regularisation
    """
    return lambda_val * np.power(np.sum(w), 2) / (m * 2)

## (if using L-2 regularisation update cost function too)




### this bit by itself is called the Frobenius Norm:
# np.power(np.sum(w), 2)


def l1_regularisation(lambda_val, m, w):
    """ aka L1 norm"""
    return (lambda_val * np.sum(np.absolute(w))) / m




## to include l2_regularisation during backpropagation we add a term:
# dW[L] = (what it would be normally) + (lambda / m)*W[L]   
## where new term is the derivative of l2_regularisation func with respect to W

# @@ Cell 42
# normalising inputs speeds up training process and gets better results, as the search space becomes 
# more symmetric (means that a single learning rate is appropriate for all dimensions with doing gradient descent)
# If data are on roughly the same scale it won't hurt to normalise inputs, but will make less of a difference


# weights increase/decrease by whatever they are to the power of the number of layers they are from the 
# output layer. So if lots of layers separate them the gradient can easily go to a very small (~0) or very 
# big number. This is the vanishing/exploding gradient. More on this:
# https://www.coursera.org/learn/deep-neural-network/lecture/C9iQO/vanishing-exploding-gradients


# something that helps exploding/vanishing network: careful weights initialisation
# the more weights you have the smaller you want individual weights to be
# to solve this, it's common to initialise random weights with:
however_many_weights = 6
n_L_minusOne = 5   # no. of nodes in preceding layer
weights_init_one_layer = np.random.randn(however_many_weights) * np.sqrt(2 / n_L_minusOne) 
                    # best to set numerator in the above to 1, not 2, if not relu
weights_init_one_layer


# there are other initialisation formulae in the vein of the above including:
# Xavier initialisation (involves using tanh())

# deciding on multiplier to use in weights initialisation (instead of 2 in the above)
# could become another hyperparameter to tune

# more on initialising weights:
# https://www.coursera.org/learn/deep-neural-network/lecture/RwqYe/weight-initialization-for-deep-networks

# @@ Cell 43
### Activation functions:
# denote generic activation function as g(x)

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1- sigmoid_output)   # * = element-wise mult; @ = matrix mult


# ReLu: better for larger values as the gradient doesn't become close to 0, which it does for tanh and sigmoid
# ReLu trains faster because of this 
def relu(x):
    return np.where(x >= 0, x, 0)

def relu_derivative(x):       
    """note x is the original input to relu, not the output from relu, unlike other
    derivatives of activation functions"""
    return np.where(x >= 0, 1, 0)

# @@ Cell 44
# load data
data = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
data = data.to_numpy()


# unique, counts = np.unique(data[:, 4], return_counts=True)
setosa_idx = data[:, 4] == 'setosa'  # simplify species column to binary
data[:, 4] = 1   # not setosa
data[setosa_idx, 4] = 0   # setosa


idx = np.random.rand(*data.shape).argsort(axis=0) # randomising order for test/train split
data = np.take_along_axis(data,idx,axis=0)


# normalise first 4 columns
for i in range(4): 
    norms_each_column = np.linalg.norm(data[:, i], axis = 0)
    data[:, i] = data[:, i] / norms_each_column

    
# np.linalg.norm calculates L2 norm, aka the sum of the squared values
"""
for i in range(4): 
    print(np.min(data[:, i]))
    print(np.max(data[:, i]))
    print(np.mean(data[:, i]))
    print(np.std(data[:, i]))
"""
    
print(*data.shape)  # return values outside tuple
print(data.shape)


# distribution of values for each column shows they are operating on the same scale with similar distributions
plt.hist(data[:, 0], alpha = 0.2)
plt.hist(data[:, 1], alpha = 0.2)
plt.hist(data[:, 2], alpha = 0.2)
plt.hist(data[:, 3], alpha = 0.2)
plt.show()

# @@ Cell 45
train_input = data[:120, :4].T   # transpose to row for each feature and column for each value: faster calcs
train_labs = data[:120, 4:].T

test_input = data[120:, :4].T
test_labs = data[120:, 4:].T

print(train_input.shape)
print(train_labs.shape)

# @@ Cell 46
def feedforward_relu_layer(X, W, B):
    """
    X = input data 
    W = input weights
    B = input bias
    
    Exports Z as it's needed for backpropagation
    """
    Z = np.dot(W, X) + B
    return relu(Z), Z  # returns 2 n * m arrays  (n = number of neurons)


def backpropagate_relu_layer(X, Z, dZ_nextLayer, W_nextLayer):
    """
    X = input at start of the layer (from input layer or previous hidden layer)
    Z = values after X been through linear calculation (yx + b)
    dZ_nextLayer = dZ from layer next-closest to output layer
    W_nextLayer = W from layer next-closest to output layer
    """
    m = X.shape[1]                                 # total input data in training set
    dZ_thisLayer = (W_nextLayer.T @ dZ_nextLayer) * relu_derivative(Z)  
    dW_this_layer = (1/m) * (dZ_thisLayer @ X.T)
    dB_this_layer = (1/m) * np.sum(dZ_thisLayer, axis=1, keepdims=True)
    return dZ_thisLayer, dW_this_layer, dB_this_layer

# @@ Cell 47
def cost_function(y, y_hat):   
    """Logistic regression cost function
    y = actual
    y_hat = predicted
    """
    m = y.shape[1]  # total predictions made
    lhs = np.dot(y, np.log(y_hat).T) # returns 1x1 array
    rhs = np.dot((1 - y), np.log(1 - y_hat).T)
    total_loss = np.sum(lhs + rhs)
    return -total_loss / m

# @@ Cell 48
def gradient_descent(X, Y, n_layer, lr, iterations):
    """
    Trains parameters for NN with 3 hidden layers, and output layer with sigmoid activation
    
    X = input table (n * m)
    Y = labels    (1 * m)
    lr = Learning rate: how much weights change on each iteration
   """
    
    loss_store = np.zeros(iterations)  # to store loss on each iteration
    
    
    # set dimension values used 
    n_0 = X.shape[0]   # total features 
    n_1 = n_layer         # number of neurons in hidden layer
    n_2 = 1         # output units: think it's 1 as only one layer. Could be 2 if linked to binary outcome somehow
    

    m = X.shape[1]   # size of input data
    
    # initialise weights             
    W_1_1 = np.random.randn(size=(n_1, n_0)) * np.sqrt(2 / n_0)  # notice weights in first layer have different dims
    W_1_2 = np.random.randn(size=(n_1, n_1)) * np.sqrt(2 / n_1) 
    W_1_3 = np.random.randn(size=(n_1, n_1)) * np.sqrt(2 / n_1) 
    B_1_1 = np.random.randn(size=(n_1, 1)) * np.sqrt(2 / n_1) 
    B_1_1 = np.random.randn(size=(n_1, 1)) * np.sqrt(2 / n_1) 
    B_1_1 = np.random.randn(size=(n_1, 1)) * np.sqrt(2 / n_1) 
    
    W_2 = np.random.randn(size=(n_2, n_1)) * np.sqrt(2 / n_1) 
    B_2 = np.random.randn(size=(n_2, 1)) * np.sqrt(2 / n_1) 

    
    """
    W_1_1 = np.random.uniform(size=(n_1, n_0)) / 100   
    W_1_2 = np.random.uniform(size=(n_1, n_1)) / 100  
    W_1_3 = np.random.uniform(size=(n_1, n_1)) / 100
    B_1_1 = np.random.uniform(size=(n_1, 1))  / 100
    B_1_2 = np.random.uniform(size=(n_1, 1))  / 100
    B_1_3 = np.random.uniform(size=(n_1, 1))  / 100
    
    W_2 = np.random.uniform(size=(n_2, n_1))  / 100
    B_2 = np.random.uniform(size=(n_2, 1))  / 100
    """
    
    # W[1] dims: n[1] * n[0]
    # B[1] dims: n[1] * 1
    # W[2] dims: n[2] * n[1]
    # B[2] dims: n[2] * 1
    
    #### starting gradient descent loop
    for i in range(iterations):


        # hidden layer: compute linear transformation of inputs
        A_1_1, Z_1_1 = feedforward_relu_layer(X, W_1_1, B_1_1)
        A_1_2, Z_1_2 = feedforward_relu_layer(A_1_1, W_1_2, B_1_2)
        A_1_3, Z_1_3 = feedforward_relu_layer(A_1_2, W_1_3, B_1_3)


        # output layer: linear transformation and activation func
        Z_2 = np.dot(W_2, A_1_3) + B_2
        Z_2 = Z_2.astype(np.float) # ensure is float; needed for sigmoid() to work
        A_2 = sigmoid(Z_2)

        
        # store cost
        loss_store[i] = cost_function(Y, A_2)


        # output layer: get derivatives
        dZ_2 = A_2 - Y                           
        dW_2 = (1/m) * (dZ_2 @ A_1_3.T)
        dB_2 = (1/m) * np.sum(dZ_2, axis=1, keepdims=True)  # sums horizontally


        # hidden layers: get derivatives 
        dZ_1_3, dW_1_3, dB_1_3 = backpropagate_relu_layer(A_1_2, Z_1_3, dZ_2, W_2)
        dZ_1_2, dW_1_2, dB_1_2 = backpropagate_relu_layer(A_1_1, Z_1_2, dZ_1_3, W_1_3)
        dZ_1_1, dW_1_1, dB_1_1 = backpropagate_relu_layer(X, Z_1_1, dZ_1_2, W_1_2)


        # update weights
        W_1_1 = W_1_1 - lr * dW_1_1
        W_1_2 = W_1_2 - lr * dW_1_2
        W_1_3 = W_1_3 - lr * dW_1_3
        B_1_1 = B_1_1 - lr * dB_1_1
        B_1_2 = B_1_2 - lr * dB_1_2
        B_1_3 = B_1_3 - lr * dB_1_3
        W_2 = W_2 - lr * dW_2
        B_2 = B_2 - lr * dB_2
        
    


    return W_1_1, W_1_2, W_1_3, B_1_1,B_1_2, B_1_3, W_2, B_2, loss_store

    

# @@ Cell 49
## training model weights
W_1_1, W_1_2, W_1_3, B_1_1,B_1_2, B_1_3, W_2, B_2, loss_store = gradient_descent(train_input, train_labs, 10, 0.02, 1000)

# @@ Cell 50
def gradient_descent(X, Y, n_layer, lr, iterations):
    """
    Trains parameters for NN with 3 hidden layers, and output layer with sigmoid activation
    
    X = input table (n * m)
    Y = labels    (1 * m)
    lr = Learning rate: how much weights change on each iteration
   """
    
    loss_store = np.zeros(iterations)  # to store loss on each iteration
    
    
    # set dimension values used 
    n_0 = X.shape[0]   # total features 
    n_1 = n_layer         # number of neurons in hidden layer
    n_2 = 1         # output units: think it's 1 as only one layer. Could be 2 if linked to binary outcome somehow
    

    m = X.shape[1]   # size of input data
    
    # initialise weights             
    W_1_1 = np.random.randn(n_1, n_0) * np.sqrt(2 / n_0)  # notice weights in first layer have different dims
    W_1_2 = np.random.randn(n_1, n_1) * np.sqrt(2 / n_1) 
    W_1_3 = np.random.randn(n_1, n_1) * np.sqrt(2 / n_1) 
    B_1_1 = np.random.randn(n_1, 1) * np.sqrt(2 / n_1) 
    B_1_1 = np.random.randn(n_1, 1) * np.sqrt(2 / n_1) 
    B_1_1 = np.random.randn(n_1, 1) * np.sqrt(2 / n_1) 
    
    W_2 = np.random.randn(n_2, n_1) * np.sqrt(2 / n_1) 
    B_2 = np.random.randn(n_2, 1) * np.sqrt(2 / n_1) 

    
    """
    W_1_1 = np.random.uniform(size=(n_1, n_0)) / 100   
    W_1_2 = np.random.uniform(size=(n_1, n_1)) / 100  
    W_1_3 = np.random.uniform(size=(n_1, n_1)) / 100
    B_1_1 = np.random.uniform(size=(n_1, 1))  / 100
    B_1_2 = np.random.uniform(size=(n_1, 1))  / 100
    B_1_3 = np.random.uniform(size=(n_1, 1))  / 100
    
    W_2 = np.random.uniform(size=(n_2, n_1))  / 100
    B_2 = np.random.uniform(size=(n_2, 1))  / 100
    """
    
    # W[1] dims: n[1] * n[0]
    # B[1] dims: n[1] * 1
    # W[2] dims: n[2] * n[1]
    # B[2] dims: n[2] * 1
    
    #### starting gradient descent loop
    for i in range(iterations):


        # hidden layer: compute linear transformation of inputs
        A_1_1, Z_1_1 = feedforward_relu_layer(X, W_1_1, B_1_1)
        A_1_2, Z_1_2 = feedforward_relu_layer(A_1_1, W_1_2, B_1_2)
        A_1_3, Z_1_3 = feedforward_relu_layer(A_1_2, W_1_3, B_1_3)


        # output layer: linear transformation and activation func
        Z_2 = np.dot(W_2, A_1_3) + B_2
        Z_2 = Z_2.astype(np.float) # ensure is float; needed for sigmoid() to work
        A_2 = sigmoid(Z_2)

        
        # store cost
        loss_store[i] = cost_function(Y, A_2)


        # output layer: get derivatives
        dZ_2 = A_2 - Y                           
        dW_2 = (1/m) * (dZ_2 @ A_1_3.T)
        dB_2 = (1/m) * np.sum(dZ_2, axis=1, keepdims=True)  # sums horizontally


        # hidden layers: get derivatives 
        dZ_1_3, dW_1_3, dB_1_3 = backpropagate_relu_layer(A_1_2, Z_1_3, dZ_2, W_2)
        dZ_1_2, dW_1_2, dB_1_2 = backpropagate_relu_layer(A_1_1, Z_1_2, dZ_1_3, W_1_3)
        dZ_1_1, dW_1_1, dB_1_1 = backpropagate_relu_layer(X, Z_1_1, dZ_1_2, W_1_2)


        # update weights
        W_1_1 = W_1_1 - lr * dW_1_1
        W_1_2 = W_1_2 - lr * dW_1_2
        W_1_3 = W_1_3 - lr * dW_1_3
        B_1_1 = B_1_1 - lr * dB_1_1
        B_1_2 = B_1_2 - lr * dB_1_2
        B_1_3 = B_1_3 - lr * dB_1_3
        W_2 = W_2 - lr * dW_2
        B_2 = B_2 - lr * dB_2
        
    


    return W_1_1, W_1_2, W_1_3, B_1_1,B_1_2, B_1_3, W_2, B_2, loss_store

    

# @@ Cell 51
## training model weights
W_1_1, W_1_2, W_1_3, B_1_1,B_1_2, B_1_3, W_2, B_2, loss_store = gradient_descent(train_input, train_labs, 10, 0.02, 1000)

# @@ Cell 52
def gradient_descent(X, Y, n_layer, lr, iterations):
    """
    Trains parameters for NN with 3 hidden layers, and output layer with sigmoid activation
    
    X = input table (n * m)
    Y = labels    (1 * m)
    lr = Learning rate: how much weights change on each iteration
   """
    
    loss_store = np.zeros(iterations)  # to store loss on each iteration
    
    
    # set dimension values used 
    n_0 = X.shape[0]   # total features 
    n_1 = n_layer         # number of neurons in hidden layer
    n_2 = 1         # output units: think it's 1 as only one layer. Could be 2 if linked to binary outcome somehow
    

    m = X.shape[1]   # size of input data
    
    # initialise weights             
    W_1_1 = np.random.randn(n_1, n_0) * np.sqrt(2 / n_0)  # notice weights in first layer have different dims
    W_1_2 = np.random.randn(n_1, n_1) * np.sqrt(2 / n_1) 
    W_1_3 = np.random.randn(n_1, n_1) * np.sqrt(2 / n_1) 
    B_1_1 = np.random.randn(n_1, 1) * np.sqrt(2 / n_1) 
    B_1_2 = np.random.randn(n_1, 1) * np.sqrt(2 / n_1) 
    B_1_3 = np.random.randn(n_1, 1) * np.sqrt(2 / n_1) 
    
    W_2 = np.random.randn(n_2, n_1) * np.sqrt(2 / n_1) 
    B_2 = np.random.randn(n_2, 1) * np.sqrt(2 / n_1) 

    
    """
    W_1_1 = np.random.uniform(size=(n_1, n_0)) / 100   
    W_1_2 = np.random.uniform(size=(n_1, n_1)) / 100  
    W_1_3 = np.random.uniform(size=(n_1, n_1)) / 100
    B_1_1 = np.random.uniform(size=(n_1, 1))  / 100
    B_1_2 = np.random.uniform(size=(n_1, 1))  / 100
    B_1_3 = np.random.uniform(size=(n_1, 1))  / 100
    
    W_2 = np.random.uniform(size=(n_2, n_1))  / 100
    B_2 = np.random.uniform(size=(n_2, 1))  / 100
    """
    
    # W[1] dims: n[1] * n[0]
    # B[1] dims: n[1] * 1
    # W[2] dims: n[2] * n[1]
    # B[2] dims: n[2] * 1
    
    #### starting gradient descent loop
    for i in range(iterations):


        # hidden layer: compute linear transformation of inputs
        A_1_1, Z_1_1 = feedforward_relu_layer(X, W_1_1, B_1_1)
        A_1_2, Z_1_2 = feedforward_relu_layer(A_1_1, W_1_2, B_1_2)
        A_1_3, Z_1_3 = feedforward_relu_layer(A_1_2, W_1_3, B_1_3)


        # output layer: linear transformation and activation func
        Z_2 = np.dot(W_2, A_1_3) + B_2
        Z_2 = Z_2.astype(np.float) # ensure is float; needed for sigmoid() to work
        A_2 = sigmoid(Z_2)

        
        # store cost
        loss_store[i] = cost_function(Y, A_2)


        # output layer: get derivatives
        dZ_2 = A_2 - Y                           
        dW_2 = (1/m) * (dZ_2 @ A_1_3.T)
        dB_2 = (1/m) * np.sum(dZ_2, axis=1, keepdims=True)  # sums horizontally


        # hidden layers: get derivatives 
        dZ_1_3, dW_1_3, dB_1_3 = backpropagate_relu_layer(A_1_2, Z_1_3, dZ_2, W_2)
        dZ_1_2, dW_1_2, dB_1_2 = backpropagate_relu_layer(A_1_1, Z_1_2, dZ_1_3, W_1_3)
        dZ_1_1, dW_1_1, dB_1_1 = backpropagate_relu_layer(X, Z_1_1, dZ_1_2, W_1_2)


        # update weights
        W_1_1 = W_1_1 - lr * dW_1_1
        W_1_2 = W_1_2 - lr * dW_1_2
        W_1_3 = W_1_3 - lr * dW_1_3
        B_1_1 = B_1_1 - lr * dB_1_1
        B_1_2 = B_1_2 - lr * dB_1_2
        B_1_3 = B_1_3 - lr * dB_1_3
        W_2 = W_2 - lr * dW_2
        B_2 = B_2 - lr * dB_2
        
    


    return W_1_1, W_1_2, W_1_3, B_1_1,B_1_2, B_1_3, W_2, B_2, loss_store

    

# @@ Cell 53
## training model weights
W_1_1, W_1_2, W_1_3, B_1_1,B_1_2, B_1_3, W_2, B_2, loss_store = gradient_descent(train_input, train_labs, 10, 0.02, 1000)

# @@ Cell 54
# viewing trend in loss
plt.plot(loss_store)

# @@ Cell 55
# applying to test set (Z matrices are created but not used)
A_1_1, Z_1_1 = feedforward_relu_layer(test_input, W_1_1, B_1_1)
A_1_2, Z_1_2 = feedforward_relu_layer(A_1_1, W_1_2, B_1_2)
A_1_3, Z_1_3 = feedforward_relu_layer(A_1_2, W_1_3, B_1_3)
Z_2 = np.dot(W_2, A_1_3) + B_2
Z_2 = Z_2.astype(np.float) # ensure is float; needed for sigmoid() to work
predictions = sigmoid(Z_2)

# @@ Cell 56
# evaluating accuracy on test set: nothing special
plt.scatter(predictions, test_labs)

# @@ Cell 57
## Notes on Practical Aspects of Deep Learning (week 1 of course 2 in deep learning specialisation)

## making a deep NN in numpy with added:
# > regularisation (drop out)
# > input normalisation
# > improved weight initialisation



## Aiming to predict whether a plant is species setosa or not

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# @@ Cell 58
W_1_1_droput = random.choices(
            population = [0, 1],
            weights = [0.2, 0.8],
            k = (5, 4)
        )


W_1_1_droput

# @@ Cell 59
W_1_1_droput = random.choices(
            population = [0, 1],
            weights = [0.2, 0.8],
            k = 20
        )


W_1_1_droput

# @@ Cell 60
W_1_1_droput = random.choices(
            population = [0, 1],
            weights = [0.2, 0.8],
            k = 20
        ).reshape(5, 4)


W_1_1_droput

# @@ Cell 61
W_1_1_droput = random.choices(
            population = [0, 1],
            weights = [0.2, 0.8],
            k = 20
        )
W_1_1_droput = np.array(W_1_1_droput)

W_1_1_droput

# @@ Cell 62
W_1_1_droput = random.choices(
            population = [0, 1],
            weights = [0.2, 0.8],
            k = 20
        )
W_1_1_droput = np.array(W_1_1_droput).reshape(5, 4)

W_1_1_droput

# @@ Cell 63
def gradient_descent(X, Y, n_layer, lr, iterations):
    """
    Trains parameters for NN with 3 hidden layers, and output layer with sigmoid activation
    
    X = input table (n * m)
    Y = labels    (1 * m)
    lr = Learning rate: how much weights change on each iteration
   """
    
    loss_store = np.zeros(iterations)  # to store loss on each iteration
    
    
    # set dimension values used 
    n_0 = X.shape[0]   # total features 
    n_1 = n_layer         # number of neurons in hidden layer
    n_2 = 1         # output units: think it's 1 as only one layer. Could be 2 if linked to binary outcome somehow
    

    m = X.shape[1]   # size of input data
    
    # initialise weights             
    W_1_1 = np.random.randn(n_1, n_0) * np.sqrt(2 / n_0)  # notice weights in first layer have different dims
    W_1_2 = np.random.randn(n_1, n_1) * np.sqrt(2 / n_1) 
    W_1_3 = np.random.randn(n_1, n_1) * np.sqrt(2 / n_1) 
    B_1_1 = np.random.randn(n_1, 1) * np.sqrt(2 / n_1) 
    B_1_2 = np.random.randn(n_1, 1) * np.sqrt(2 / n_1) 
    B_1_3 = np.random.randn(n_1, 1) * np.sqrt(2 / n_1) 
    
    W_2 = np.random.randn(n_2, n_1) * np.sqrt(2 / n_1) 
    B_2 = np.random.randn(n_2, 1) * np.sqrt(2 / n_1) 

    
    """
    W_1_1 = np.random.uniform(size=(n_1, n_0)) / 100   
    W_1_2 = np.random.uniform(size=(n_1, n_1)) / 100  
    W_1_3 = np.random.uniform(size=(n_1, n_1)) / 100
    B_1_1 = np.random.uniform(size=(n_1, 1))  / 100
    B_1_2 = np.random.uniform(size=(n_1, 1))  / 100
    B_1_3 = np.random.uniform(size=(n_1, 1))  / 100
    
    W_2 = np.random.uniform(size=(n_2, n_1))  / 100
    B_2 = np.random.uniform(size=(n_2, 1))  / 100
    """
    
    # W[1] dims: n[1] * n[0]
    # B[1] dims: n[1] * 1
    # W[2] dims: n[2] * n[1]
    # B[2] dims: n[2] * 1
    
    #### starting gradient descent loop
    for i in range(iterations):
        
        
         
        # create matrices for random dropout for first two hidden layers
        W_1_1_droput = random.choices(
            population = [0, 1.25],     # is 1.25 as need expected value to remain same 
            weights = [0.2, 0.8],
            k = n_1 * n_0
        )
        W_1_1_droput = np.array(W_1_1_droput).reshape(n_1, n_0)
        
        
        W_1_2_droput = random.choices(
            population = [0, 1.25],     # is 1.25 as need expected value to remain same 
            weights = [0.2, 0.8],
            k = n_1 * n_1
        )
        W_1_2_droput = np.array(W_1_2_droput).reshape(n_1, n_1)

        
        
        
        
        # applying dropout matrices to weights
        W_1_1_dropped_out = W_1_1 * W_1_1_droput
        W_1_2_dropped_out = W_1_2 * W_1_2_droput
        
        


        # hidden layer: compute linear transformation of inputs
        A_1_1, Z_1_1 = feedforward_relu_layer(X, W_1_1_dropped_out, B_1_1)
        A_1_2, Z_1_2 = feedforward_relu_layer(A_1_1, W_1_2_dropped_out, B_1_2)
        A_1_3, Z_1_3 = feedforward_relu_layer(A_1_2, W_1_3, B_1_3)


        # output layer: linear transformation and activation func
        Z_2 = np.dot(W_2, A_1_3) + B_2
        Z_2 = Z_2.astype(np.float) # ensure is float; needed for sigmoid() to work
        A_2 = sigmoid(Z_2)

        
        # store cost
        loss_store[i] = cost_function(Y, A_2)


        # output layer: get derivatives
        dZ_2 = A_2 - Y                           
        dW_2 = (1/m) * (dZ_2 @ A_1_3.T)
        dB_2 = (1/m) * np.sum(dZ_2, axis=1, keepdims=True)  # sums horizontally


        # hidden layers: get derivatives 
        dZ_1_3, dW_1_3, dB_1_3 = backpropagate_relu_layer(A_1_2, Z_1_3, dZ_2, W_2)
        dZ_1_2, dW_1_2, dB_1_2 = backpropagate_relu_layer(A_1_1, Z_1_2, dZ_1_3, W_1_3)
        dZ_1_1, dW_1_1, dB_1_1 = backpropagate_relu_layer(X, Z_1_1, dZ_1_2, W_1_2_dropped_out)


        # update weights
        W_1_1 = W_1_1 - lr * dW_1_1
        W_1_2 = W_1_2 - lr * dW_1_2
        W_1_3 = W_1_3 - lr * dW_1_3
        B_1_1 = B_1_1 - lr * dB_1_1
        B_1_2 = B_1_2 - lr * dB_1_2
        B_1_3 = B_1_3 - lr * dB_1_3
        W_2 = W_2 - lr * dW_2
        B_2 = B_2 - lr * dB_2
        
    


    return W_1_1, W_1_2, W_1_3, B_1_1,B_1_2, B_1_3, W_2, B_2, loss_store

    

# @@ Cell 64
## training model weights
W_1_1, W_1_2, W_1_3, B_1_1,B_1_2, B_1_3, W_2, B_2, loss_store = gradient_descent(train_input, train_labs, 10, 0.02, 1000)

# @@ Cell 65
# viewing trend in loss
plt.plot(loss_store)

# @@ Cell 66
# applying to test set (Z matrices are created but not used)
A_1_1, Z_1_1 = feedforward_relu_layer(test_input, W_1_1, B_1_1)
A_1_2, Z_1_2 = feedforward_relu_layer(A_1_1, W_1_2, B_1_2)
A_1_3, Z_1_3 = feedforward_relu_layer(A_1_2, W_1_3, B_1_3)
Z_2 = np.dot(W_2, A_1_3) + B_2
Z_2 = Z_2.astype(np.float) # ensure is float; needed for sigmoid() to work
predictions = sigmoid(Z_2)

# @@ Cell 67
# evaluating accuracy on test set: seems to perform very slightly less awfully
plt.scatter(predictions, test_labs)

# @@ Cell 68
## training model weights
W_1_1, W_1_2, W_1_3, B_1_1,B_1_2, B_1_3, W_2, B_2, loss_store = gradient_descent(train_input, train_labs, 10, 0.02, 10000)

# @@ Cell 69
# viewing trend in loss: dropout stops it being monotonic
plt.plot(loss_store)

# @@ Cell 70
# applying to test set (Z matrices are created but not used)
A_1_1, Z_1_1 = feedforward_relu_layer(test_input, W_1_1, B_1_1)
A_1_2, Z_1_2 = feedforward_relu_layer(A_1_1, W_1_2, B_1_2)
A_1_3, Z_1_3 = feedforward_relu_layer(A_1_2, W_1_3, B_1_3)
Z_2 = np.dot(W_2, A_1_3) + B_2
Z_2 = Z_2.astype(np.float) # ensure is float; needed for sigmoid() to work
predictions = sigmoid(Z_2)

# @@ Cell 71
# evaluating accuracy on test set: 
# performs very slightly less awfully with input normalisation and better weights initialisation
# then goes back to it's mediocre self with dropout added
plt.scatter(predictions, test_labs)

# @@ Cell 72
predictions      # very close to 66.7% for all values

