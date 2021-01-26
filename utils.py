import numpy as np
import os, sys
from sklearn.decomposition import PCA

# context model from the paper - noisy drift applied to spherical coordinates
def fast_n_sphere(n_steps=20, dim=10, var=0.25, mean=0.25):
    # initialize the spherical coordinates to ensure each context run
    # begins in a new random location on the unit sphere
    ros = np.random.random(dim - 1)
    slen = n_steps
    ctxt = np.zeros((slen, dim))
    for i in range(slen):
        noise = np.random.normal(mean, var, size=(dim - 1)) # add a separately-drawn Gaussian to each spherical coord
        ros += noise
        ctxt[i] = convert_spherical_to_angular(dim, ros)
    if mean == 0:
        stem = "new_spherical_diffusion_"
    else:
        stem = "new_spherical_drift_"
    fn_name = stem + "d=" + str(dim) + "_m=" + str(mean) + "_v=" + str(var)
    return ctxt, fn_name

# Convert spherical coordinates to angular ones
def convert_spherical_to_angular(dim, ros):
    ct = np.zeros(dim)
    ct[0] = np.cos(ros[0])
    prod = np.product([np.sin(ros[k]) for k in range(1, dim - 1)])
    n_prod = prod
    for j in range(dim - 2):
        n_prod /= np.sin(ros[j + 1])
        amt = n_prod * np.cos(ros[j + 1])
        ct[j + 1] = amt
    ct[dim - 1] = prod
    return ct

def return_sphere_context_supplier_from_params(d, m, v):
    return lambda idx, n_steps: fast_n_sphere(n_steps, d, v, m)

# instead of redrawing the context every epoch, 
# make a really long context walk and draw from different points on it
def long_walk(long_context, idx=0, n_steps=20):
  n_idx = idx % int(len(long_context) / n_steps - 1)
  ctxt = long_context[n_idx * n_steps: (n_idx + 1) * n_steps]
  if len(ctxt) < n_steps:
    raise ValueError(
      "Context walk was too short, with idx " + str(idx) + " using n_idx " + str(n_idx) + " and len " + str(len(long_context)))
  return ctxt

