
# Learned from: https://www.youtube.com/watch?v=cXHvC_FGx24

# Optimisation: Find input values which minimise a function, subject to 
# constraints on those inputs, and with best-guess starting inputs

# Different optimisations, or optimising via Monte Carlo simulations, will
# get you different weights. It's not obvious which is best up front


from scipy.optimize import minimize, basinhopping
import numpy as np


def objective(x):
	"""
	The function you're aiming to minimise, with 3 values
	"""
	return x[0] * (x[1] + x[2])


def constraint1(x):
	return ((x[2]*0) + x[0]*x[1]) > 2


def constraint2(x):
	return ((x[2]*0) + 1 + x[0]*x[1]) > 3


x0 = 1, 4, 5    # initial best guess

bounds = [(0, 10), (-10, 100), (-10, 100)]  # upper and lower bounds to explore for each input

con1 = {'type':'ineq', 'fun':constraint1}   # add constraints, saying it's an "inequality" type constraint
con2 = {'type':'ineq', 'fun':constraint2}
cons = [con1, con2]


# Get errors with some methods, however 'Powell' runs ok
solution = minimize(objective, x0, method='Powell', bounds=bounds, constraints=cons)
print(solution)
print(solution.x)  # solution values



# Might send some values to infinity
solution = minimize(objective, x0, method='Powell')
print('solution_no_bounds_or_constraints')
print(solution)





### basinhopping is essentially simulated annealing
minimizer_kwargs = {"method":"Powell", "jac":False}
solution = basinhopping(objective, x0, minimizer_kwargs=minimizer_kwargs) #,niter=10)

print(f'solution: {solution}')



# scipy.optimize.root exists to find the root of functions (where their lines cross the x axis)











