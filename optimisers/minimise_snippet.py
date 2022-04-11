import scipy

x = {some array input
y =  {target array input

def model_to_apply(x, sensitivity_parameter):
    unscaled_proportion_trips_made = x * np.exp(sensitivity_parameter * x)
    return unscaled_proportion_trips_made / np.sum(unscaled_proportion_trips_made)

def loss_function(x, y, sensitivity_parameter): return np.sum((y - model_to_apply(x, sensitivity_parameter))**2)

bounds = [(None, None)]  # no bounds

# find minimal coefs
res = scipy.optimize.minimize(lambda coeffs: loss_function(x, y, *coeffs), x0=np.zeros(1), bounds=bounds)


# view fit 
sensitivity_parameter = res.x[0]
