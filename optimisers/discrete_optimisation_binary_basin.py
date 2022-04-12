### basin hopping in scipy, deciding optimal values of binary variables

from scipy.optimize import basinhopping
import numpy as np


# set actual data
dummy_table = np.asarray([[30, 50, 60], [50, 20, 21], [43, 12, 43]])
targets = np.asarray([100, 70, 22])


# set initial values
x0 = np.asarray([1, 0, 0])   



# function to minimise
def get_min(x0):
    
    x0 = np.rint(x0) # round from 0-1 float to binary 0 or 1
    
    new_array = np.asarray(dummy_table) * x0
    
    row_sums = np.sum(new_array, axis = 1)
    
    squared_loss = np.sum((row_sums - targets) ** 2)
    
    #print(squared_loss)
    
    return squared_loss
    
    
    
    
def callback(x, f, accept):
    """
    callback func is passed 3 inputs from inside the basinhopping func
    
    called on each iter **I think**. But sometimes not. So not clear when triggered. 
    
    Set "return True" to end optimisation process
    """
    print(f'x (current params): {x}\nf (loss): {f}\naccept (if new params are accepted): {accept}')
    
    global iter
    print(f'iter is {iter}')
    iter += 1

    
    
        
constraints = [(0,1)] * len(targets)    # set 0-1 range for all inputs

minimizer_kwargs = {"method": "L-BFGS-B",     # optimisation methods
                        "bounds":constraints,
                        "options":{
                            "eps": 0.05
                            }
                       }
    
iter = 0
results = basinhopping(get_min, x0, 
                       minimizer_kwargs=minimizer_kwargs, 
                       niter = 10_000, 
                       niter_success = 100,  # if current minimum isnt improved on after this many iters
                       callback=callback  # run this on each iteration
                      )

print(f'Found parameters of {np.rint(results.x)}')

results
