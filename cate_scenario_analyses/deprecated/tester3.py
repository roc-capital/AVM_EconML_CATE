import pymc as pm
import numpy as np

with pm.Model() as m:
    mu = pm.Normal("mu", 0, 1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=np.random.randn(100))
    idata = pm.sample(draws=200, tune=200, chains=2, cores=1, progressbar=True)
