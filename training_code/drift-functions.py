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

# instead of redrawing the context every epoch, make a really long context walk and draw from different points on it
def long_walk(long_context, idx=0, n_steps=20):
  n_idx = idx % int(len(long_context) / n_steps - 1)
  ctxt = long_context[n_idx * n_steps: (n_idx + 1) * n_steps]
  if len(ctxt) < n_steps:
    raise ValueError(
      "Context walk was too short, with idx " + str(idx) + " using n_idx " + str(n_idx) + " and len " + str(len(long_context)))
  return ctxt


# DEPRECATED - the context models below were experiments before we landed on our current context model

def normed_iid_noise_20():
    slen = 20
    cdim = 100
    ctxt = np.zeros((cdim, slen))
    noise_scale = 0.10
    for i in range(1, slen):
        noise = np.random.normal(0, noise_scale, size=cdim)
        noised = ctxt[:, i - 1] - noise
        ctxt[:, i] = noised / np.linalg.norm(noised)
    fn_name = "normed_iid_noise_cdim" + str(cdim) + "_sdim20_delta" + str(noise_scale)
    ctxt = [list(ctxt[:, i]) for i in range(slen)]
    return ctxt, fn_name

def no_noise(ctxts):
    return ctxts, "no_noise"

def continuous_ctxts_30():
    ctxts = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7,7], [8,8], [9,9], [10,10], [11,11], [12,12], [13,13],
             [14,14], [15,15], [16,16], [17,17], [18,18], [19,19], [20,20], [21, 21], [22, 22], [23, 23], [24, 24],
             [25, 25], [26, 26], [27, 27], [28, 28], [29, 29], [30, 30]]
    return ctxts, "relu_nonoise_context30"

def continuous_ctxts_10():
    ctxts = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10]]
    return ctxts, "relu_nonoise_context10"

def continuous_ctxts_6():
    ctxts = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
    return ctxts, "relu_nonoise_context6"

def gaussian_noise():
    ctxts = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12],
             [13, 13],
             [14, 14], [15, 15], [16, 16], [17, 17], [18, 18], [19, 19], [20, 20]]
    mean = 0.0
    var = 0.2
    noised_ctxts = [None] * len(ctxts)
    if ctxts != []:
        noise = np.random.normal(mean, var, (len(ctxts[0]), len(ctxts)))
        for i in range(len(ctxts)):
            noised_ctxts[i] = list(ctxts[i] + noise[:,i])
    return noised_ctxts, "new_gaussian_m=" + str(mean) + "_sd=" + str(var)

def dense_integrated_stim_context(stims):
    slen = 20
    cdim = len(stims[0])
    ctxt = np.zeros((cdim, slen))
    noise_scale = 0.10
    for i in range(1, slen):
        noise = np.random.normal(0, noise_scale, size=cdim)
        ctxt[:, i] = ctxt[:, i - 1] - noise
    fn_name = "iid_noise_cdim" + str(cdim) + "_sdim20_delta" + str(noise_scale)
    ctxt = [list(ctxt[:, i]) for i in range(slen)]
    return ctxt, fn_name

def context_integration_diffusion_old():
    n_steps = 20
    var = 0.1
    mean = 0
    stim_d = 100
    drift_param = 0.1
    decay_rate = 0.1
    normed = True
    ftc_mat = np.identity(stim_d)
    ctxt = np.zeros((n_steps, stim_d))
    ctxt[0,0] = 1.0
    for i in range(1, n_steps):
        stim = np.random.normal(mean, var, size=(stim_d))
        stim = np.expand_dims(stim, axis=0)
        # print((np.matmul(stim, ftc_mat)).shape)
        c_in = (np.matmul(stim, ftc_mat))
        # print(c_in.shape)
        ctxt[i] = (1 - decay_rate) * ctxt[i - 1] + drift_param * c_in
        if normed:
            ctxt[i] = ctxt[i] / np.linalg.norm(ctxt[i])

def context_integration_diffusion():
    n_steps = 20
    var = 0.1
    mean = 0.0
    beta = 0.25
    stim_d = 100
    ftc_mat = np.identity(stim_d)
    ctxt = np.zeros((n_steps, stim_d))
    ctxt[0,0] = 1.0
    for i in range(1, n_steps):
        stim = np.random.normal(mean, var, size=(stim_d))
        stim = np.expand_dims(stim, axis=0)
        c_in = (np.matmul(stim, ftc_mat))
        c_dot = np.dot(ctxt[i - 1], c_in[0])

        p = np.sqrt(1 + beta**2 * (c_dot ** 2 - 1)) - beta * c_dot
        ctxt[i] = p * ctxt[i - 1] + beta * c_in
    ctxt = [list(ctxt[i]) for i in range(n_steps)]
    fn_name = "gaussian_feature_integration_d=" + str(stim_d) + "_v=" + str(var) + "_beta=" + str(beta)
    return ctxt, fn_name

def context_integration_diffusion_varmult(stims, n_steps=20):
    var = 0.5
    mean = 0.0
    beta = 0.1
    stim_d = 100
    ftc_mat = np.identity(stim_d)
    ctxt = np.zeros((n_steps, stim_d))
    ctxt[0, 0] = 1.0
    for i in range(1, n_steps):
        stim = np.random.normal(mean, var, size=(stim_d))
        stim = np.expand_dims(stim, axis=0)
        c_in = (np.matmul(stim, ftc_mat))
        c_dot = np.dot(ctxt[i - 1], c_in[0])
        p = np.sqrt(1 + beta**2 * (c_dot ** 2 - 1)) - beta * c_dot
        ctxt[i] = p * ctxt[i - 1] + beta * c_in
        ctxt[i] = ctxt[i] / np.linalg.norm(ctxt[i])
    ctxt = [list(ctxt[i]) for i in range(n_steps)]
    fn_name = "gf_integration_normed_noutput=" + str(stim_d) + "_v=" + str(var) + "_beta=" + str(beta)
    return ctxt, fn_name

def context_integration_diffusion_multistep(stims, n_steps=20):
    var = 0.1
    mean = 0.0
    beta = 0.3
    stim_d = 100
    ftc_mat = np.identity(stim_d)
    ctxt = np.zeros((n_steps, stim_d))
    ctxt[0, 0] = 1.0
    steps_per = 1
    for i in range(1, n_steps):
        prevc = ctxt[i - 1]
        for j in range(steps_per):
            stim = np.random.normal(mean, var, size=(stim_d))
            stim = np.expand_dims(stim, axis=0)
            c_in = (np.matmul(stim, ftc_mat))
            c_dot = np.dot(prevc, c_in[0])
            p = np.sqrt(1 + beta**2 * (c_dot ** 2 - 1)) - beta * c_dot
            prevc = p * prevc + beta * c_in
            prevc = prevc / np.linalg.norm(prevc)
        ctxt[i] = prevc
    ctxt = [list(ctxt[i]) for i in range(n_steps)]
    fn_name = "cint_diffusion_multistep=" + str(stim_d) + "_v=" + str(var) + "_beta=" + str(beta) + "_steps=" + str(steps_per)
    return ctxt, fn_name


def cint_diff_varying_var_multistep(stims, n_steps=10, var_props=[(0.01, 0.4), (0.05, 0.3), (0.1, 0.2), (0.3, 0.1)], mean=0, stim_d=100, beta=0.3, multi_steps=10):
    if np.abs(np.sum([pair[1] for pair in var_props]) - 1.0) > 0.001:
        raise ValueError("Proportions must sum to 1 but got " + str(np.sum([pair[1] for pair in var_props])))
    ftc_mat = np.identity(stim_d)
    ctxt = np.zeros((n_steps, stim_d))
    ctxt[0,0] = 1.0
    var_inds = {}
    prev_idx = 0
    for (var, prop) in var_props:
        new_idx = prev_idx + int(prop * stim_d)
        if new_idx == stim_d - 1:
            new_idx = stim_d
        var_inds[var] = list(range(prev_idx, new_idx))
        prev_idx = new_idx
    for i in range(1, n_steps):
        prev_c = ctxt[i - 1]
        for j in range(multi_steps):
            stim = np.zeros(stim_d)
            for var, inds in var_inds.items():
                noise = np.random.normal(mean, float(var), size=(len(inds)))
                stim[inds[0]:inds[-1] + 1] = noise
            stim = np.expand_dims(stim, axis=0)
            c_in = (np.matmul(stim, ftc_mat))
            c_dot = np.dot(prev_c, c_in[0])
            p = np.sqrt(1 + beta**2 * (c_dot ** 2 - 1)) - beta * c_dot
            prev_c = p * prev_c + beta * c_in
            prev_c = prev_c / np.linalg.norm(prev_c)
        ctxt[i] = prev_c
    ctxt = [list(ctxt[i]) for i in range(n_steps)]
    fn_name = "multistep_multivariance_d=" + str(stim_d) + "_v=" + str(var_props) + "_beta=" + str(beta) + "_multisteps=" + str(multi_steps)
    return ctxt, fn_name

def simple_context_integration_diffusion_multistep(stims, n_steps=10):
    var = 0.1
    mean = 0
    stim_d = 100
    beta = 0.1
    multi_steps = 1
    ctxt = np.zeros((n_steps, stim_d))
    ctxt[0,0] = 1.0
    for i in range(1, n_steps):
        prev_c = ctxt[i - 1]
        for j in range(multi_steps):
            stim = np.random.normal(mean, var, size=(stim_d))
            stim = np.expand_dims(stim, axis=0)
            prev_c = prev_c + beta * stim
            # print(np.linalg.norm(prev_c))
            prev_c = prev_c / np.linalg.norm(prev_c)
        ctxt[i] = prev_c
    ctxt = [list(ctxt[i]) for i in range(n_steps)]
    fn_name = "simple_integration_d=" + str(stim_d) + "_v=" + str(var) + "_beta=" + str(beta) + "_multisteps=" + str(multi_steps)
    return ctxt, fn_name


def simple_scaled_context_integration_diffusion_multistep(n_steps=10, var=0.2, mean=0.0, dim=10, beta=0.2, multi_steps=1):
    ctxt = np.zeros((n_steps, dim))
    ctxt[0] = np.random.random(dim) * np.random.randint(0, 10)
    for i in range(1, n_steps):
        prev_c = ctxt[i - 1]
        for j in range(multi_steps):
            stim = np.random.normal(mean, var, size=(dim))
            stim = np.expand_dims(stim, axis=0)
            prev_c = (1 - beta) * prev_c + beta * stim
            prev_c = prev_c / np.linalg.norm(prev_c)
        ctxt[i] = prev_c
    fn_name = "simple_scaled_integration_d=" + str(dim) + "_m=" + str(mean) + "_v=" + str(var) + "_beta=" + str(beta) + "_multisteps=" + str(multi_steps)
    ctxt = [list(ctxt[i]) for i in range(n_steps)]
    return ctxt, fn_name

def unnormed_context_integration_diffusion_multistep(n_steps=10, var=0.2, mean=0.0, dim=10, beta=0.5, multi_steps=1, idx=0):
    ctxt = np.zeros((n_steps, dim))
    ctxt[0] = np.random.random(dim) * np.random.randint(0, 10)
    for i in range(1, n_steps):
        prev_c = ctxt[i - 1]
        for j in range(multi_steps):
            stim = np.random.normal(mean, var, size=(dim))
            stim = np.expand_dims(stim, axis=0)
            prev_c = (1 - beta) * prev_c + beta * stim
        ctxt[i] = prev_c
    fn_name = "unnormed_integration_d=" + str(dim) + "_m=" + str(mean) + "_v=" + str(var) + "_beta=" + str(beta) + "_multisteps=" + str(multi_steps)
    ctxt = [list(ctxt[i]) for i in range(n_steps)]
    return ctxt, fn_name

def context_integration_diffusion_longwalk(stims, idx=0, n_steps=20):
    n_idx = idx % int(len(long_context) / n_steps - 1)
    ctxt = long_context[n_idx * n_steps: (n_idx + 1) * n_steps]
    if len(ctxt) < n_steps:
        raise ValueError("too short with idx " + str(idx) + " using n_idx " + str(n_idx) + " and len " + str(len(long_context)))
    fn_name = long_fn_name + "_longwalk"
    return ctxt, fn_name

def andre_context_integration_diffusion_longwalk(stims, idx=0, n_steps=20):
    n_idx = idx % int(len(long_context) / n_steps - 1)
    ctxt = long_context[n_idx * n_steps: (n_idx + 1) * n_steps]
    if len(ctxt) < n_steps:
        raise ValueError("too short with idx " + str(idx) + " using n_idx " + str(n_idx) + " and len " + str(len(long_context)))
    fn_name = long_fn_name + "_longwalk"
    return ctxt


def dim_reduced_diffusion(stims, n_steps=1000, var=0.5, h_dim=1000, l_dim=100):
    activation = np.zeros((n_steps, h_dim))
    activation[0,0] = 1.0
    for i in range(1, n_steps):
        noise = np.random.normal(0.0, var, size=h_dim)
        activation[i] = activation[i - 1] - noise
    pca = PCA(n_components=l_dim)
    components = pca.fit_transform(activation)
    fn_name = "dim_reduced_diffusion_" + "diffd=" + str(h_dim) + "_ctxtd=" + str(l_dim) + "_v=" + str(var)
    return components, fn_name

def andre_drift(n_steps=1000,var=0.2,mean=1.0, beta=0.0, dim=100):
  """
  every element of initial c_t set to anywhere within [0,1]
  steps kept constant
  """
  context_drift = -np.ones([n_steps,dim])
  c_t = np.random.random(dim) # * np.random.randint(0, 10)
  # c_t = np.ones(cdim)
  for step in range(n_steps):
    delta_t = np.random.normal(mean,var,dim)
    # if var != 0:
    delta_t /= np.linalg.norm(delta_t)
    c_t += delta_t
    context_drift[step] = c_t
  fn_name = "andre_context_d=" + str(dim) + "_m=" + str(mean) + "_v=" + str(var)
  ctxt = [list(context_drift[i]) for i in range(n_steps)]
  return ctxt, fn_name

def iid_noise(n_steps=20, dim=10, var=0.0, mean=1):
    ctxt = np.zeros((n_steps, dim))
    ctxt[0] = np.random.random(dim)
    for i in range(1, n_steps):
        noise = np.random.normal(mean, var, size=dim)
        ctxt[i] = ctxt[i - 1] + noise
    ctxt = [list(ctxt[i]) for i in range(n_steps)]
    if mean == 0:
        stem = "walk_diffusion_"
    else:
        stem = "walk_drift_"
    fn_name = stem + "d=" + str(dim) + "_m=" + str(mean) + "_v=" + str(var)
    return ctxt, fn_name

def brownian_linear_diffusion_20():
    # The Wiener process parameter.
    delta = 0.25
    # Total time.
    T = 10.0
    # Number of steps.
    N = 20
    # Time step size
    dt = T / N
    # Initial values of x.
    cdim = 5
    x = np.empty((cdim, N + 1))
    x[:, 0] = 0.0

    brownian(x[:, 0], N, dt, delta, out=x[:, 1:])
    ctxts = [list(x[:,i]) for i in range(1, N+1)]
    fn_name = "brownian_linear_drift_cdim" + str(cdim) + "_sdim20"
    return ctxts, fn_name

def brownian_sphere_drift_ctxts_20():
    global sphere_idx
    ctxt = [list(step) for step in sphere_simdata[sphere_idx:(sphere_idx + 20)]]
    sphere_idx += 20
    return ctxt, "brownian_sphere_drift_20"

def arc_drift_ctxts():
    slen = 20
    theta = 2 * np.pi * np.random.random()
    phi = 2 * np.pi * np.random.random()
    theta_d = 2 * np.pi / (slen * 2)
    ctxts = np.zeros((slen, 3))
    for i in range(slen):
        v = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        theta += theta_d
        ctxts[i] = v
    ctxts = [list(step) for step in ctxts]
    return ctxts, "arc_drift_20"

def noisy_arc_drift_ctxts():
    slen = 20
    noise_sf = 2 * np.pi / 20
    theta = 2 * np.pi * np.random.random()
    phi = 2 * np.pi * np.random.random()
    theta_d = 2 * np.pi / (slen * 2)
    phi_d = 2 * np.pi / slen / 2
    ctxts = np.zeros((slen, 3))
    for i in range(slen):
        v = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        theta += theta_d + ((np.random.random() - 0.5) * noise_sf)
        ctxts[i] = v
    ctxts = [list(step) for step in ctxts]
    return ctxts, "noisy_arc_drift_20"

def old_uniform_sphere_drift():
    slen = 20
    theta = 2 * np.pi * np.random.random()
    phi = 2 * np.pi * np.random.random()
    beta = 2 * np.pi / slen
    ax = np.array([0, 0, 1])
    ctxts = np.zeros((slen, 3))
    for i in range(slen):
        v = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        ctxts[i] = v
        b = np.cos(beta) * v + np.sin(beta) * (ax * v) + (np.dot(ax, v)) * (1 - np.cos(beta)) * v
        theta = np.arctan(np.sqrt(b[0]**2 + b[1]**2) / b[2])
        phi = np.arctan(b[1] / b[0])
    ctxts = [list(step) for step in ctxts]
    return ctxts, "uniform_sphere_drift_20"

def uniform_sphere_drift_ctxt_20():
    slen = 20
    theta = 2 * np.pi * np.random.random()
    phi = 2 * np.pi * np.random.random()
    beta = 2 * np.pi / (slen * 2)
    ctxts = np.zeros((slen, 3))
    for i in range(slen):
        v = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        theta += beta
        phi += beta
        ctxts[i] = v
    ctxts = [list(step) for step in ctxts]
    return ctxts, "uniform_sphere_ctxt_20"

def path_integration(n_steps=10, var=0.3, mean=0.0, dim=10, beta=0.3, multi_steps=1, normed=False):
    ctxt = np.zeros((n_steps, dim))
    ctxt[0] = np.random.random(dim)
    for i in range(1, n_steps):
        prev_c = ctxt[i - 1]
        for j in range(multi_steps):
            stim = np.random.normal(mean, var, size=(dim))
            stim = np.expand_dims(stim, axis=0)
            prev_c = (1 - beta) * prev_c + beta * stim
            # print(np.linalg.norm(prev_c))
            if normed:
                prev_c = prev_c / np.linalg.norm(prev_c)
        ctxt[i] = prev_c
    if mean == 0:
        stem = "path_integration_diffusion_"
    else:
        stem = "path_integration_drift_"
    fn_name = stem + "d=" + str(dim) + "_m=" + str(mean) + "_v=" + str(var) + "_b=" + str(beta)
    return ctxt, fn_name

def n_sphere(n_steps=20, dim=10, var=0.25, mean=0.25, beta=0.0):
    ros = np.random.random(dim - 1)
    slen = n_steps
    ctxt = np.zeros((slen, dim))
    for i in range(slen):
        noise = np.random.normal(mean, var, size=(dim - 1))
        ros += noise
        ct = np.zeros(dim)
        ct[0] = np.cos(ros[0])
        for j in range(dim - 2):
            amt = np.product([np.sin(ros[k]) for k in range(j + 1)])
            amt *= np.cos(ros[j + 1])
            ct[j + 1] = amt
        ct[dim - 1] = np.product([np.sin(ros[j]) for j in range(dim - 1)])
        ctxt[i] = ct
    ctxt = ctxt.T
    ctxt = [list(ctxt[:, i]) for i in range(slen)]
    if mean == 0:
        stem = "spherical_diffusion_"
    else:
        stem = "spherical_drift_"
    fn_name = stem + "d=" + str(dim) + "_m=" + str(mean) + "_v=" + str(var)
    return ctxt, fn_name


def multiscale_n_sphere(n_steps=20, dim=10, var=0.25, mean=0.25, beta=0.0):
    ros = np.zeros(dim - 1)
    # ros = np.zeros(dim - 1)
    slen = n_steps
    ctxt = np.zeros((slen, dim))
    ms = np.linspace(0, mean, num=dim)
    ms = ms[1:]
    # should this be 0 variance? Or variance that increases along w mean of dimension?
    vs = np.zeros(dim - 1)
    for i in range(slen):
        noise = np.random.normal(ms, vs, size=(dim - 1))
        ros += noise
        ct = np.zeros(dim)
        prod = 1
        for j in range(dim - 1):
            amt = prod * np.cos(ros[j])
            ct[j] = amt
            prod *= np.sin(ros[j])
        ct[dim - 1] = prod
        ctxt[i] = ct
    if mean == 0:
        stem = "multiscale_spherical_diffusion_"
    else:
        stem = "multiscale_spherical_drift_"
    fn_name = stem + "d=" + str(dim) + "_m=" + str(mean) + "_v=" + str(var)
    return ctxt, fn_name