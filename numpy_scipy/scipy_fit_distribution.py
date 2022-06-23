from scipy.stats import beta, rv_continuous
import pandas as pd
import numpy as np

# generate beta dist
a, b = 8., 4.
x = beta.rvs(a, b, size=1000)


# fit a beta curve to do, exporting fitted parameters
a1, b1, loc1, scale1 = beta.fit(x)


# Scipy has over 70 distributions one can fit to:
# https://docs.scipy.org/doc/scipy/reference/stats.html
