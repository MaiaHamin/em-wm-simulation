import numpy as np

# model of CR as spherical coordinates updated by a noisy drift
def spherical_drift(n_steps=20, dim=10, var=0.25, mean=0.25):
    # initialize the spherical coordinates to ensure each context run begins in a new random location on the unit sphere
    ros = np.random.random(dim - 1)
    slen = n_steps
    ctxt = np.zeros((slen, dim))
    for i in range(slen):
        noise = np.random.normal(mean, var, size=(dim - 1)) # add a separately-drawn Gaussian to each spherical coord
        ros += noise
        ctxt[i] = convert_spherical_to_angular(dim, ros)
    return ctxt

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
    return lambda idx, n_steps: spherical_drift(n_steps, d, v, m)

# Generate identity matrix where each row is a one-hot representation of a stimulus
def gen_stims(sdim):
    return np.identity(sdim)

# Flip bits on an input vector with probability pr_flip to mimic processing noise / distortion -- unused
def bit_flip_noise(vector, pr_flip):
    for ind in range(len(vector)):
        if np.random.uniform(0, 1) < pr_flip:
            if (vector[ind] > 0.5):
                vector[ind] = vector[ind] - 1
            else:
                vector[ind] = vector[ind] + 1
    return vector

