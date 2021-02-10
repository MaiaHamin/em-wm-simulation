import numpy as np
import os, sys
from sklearn.decomposition import PCA
from scipy.spatial import distance 

"""
EXPERIMENT/SIMULATION CODE 

# ss = 2
# get_nback_idxs(ss,1)
# get_sternberg_idxs(ss)
# sim_context_dist(5,get_sternberg_idxs)
# sim_context_dist(3,get_nback_idxs)

"""

EXP_LEN = 25
NBACK_DELTA = 1

def get_nback_idxs(set_size,delta,verb=False):
  """
  - (o) current context    (exp_len-1)
  - (i)  target context    (nback)
  - (ii) lure context (nback+-k)
  """
  current_t = EXP_LEN - 1
  target_t = current_t - set_size
  lure_t = target_t+delta
  if verb: print('ss',set_size,'current_t',current_t,
            'target',target_t,'nontarget',lure_t)
  return [current_t],[target_t],[lure_t]

def get_sternberg_idxs(set_size,delta=None,verb=False):
  """
  - (o) current context    (2*trial_len)
    - context of probe of second list
  - (i)  target contexts    (current list)
  - (ii) lure context (previous list)
  """
  trial_len = set_size + 1 # study list + probe
  current_context_idx = (2*trial_len)-1 
  # NB previous probes not included in oldL 
  # but could theoretically contribute to PI
  list1_begin,list1_end = 0,trial_len-1
  list2_begin,list2_end = trial_len,(2*trial_len)-1
  # using range of idxs to simulate full lists
  target_list_idxs = np.arange(list1_begin,list1_end)
  lure_list_idxs = np.arange(list2_begin,list2_end)
  #
  return [current_context_idx],target_list_idxs,lure_list_idxs



def sim_context_dist(set_size,get_idxs_fn,
  delta=NBACK_DELTA,nitr=99,metric='cosine',
  cmean=.3,cvar=.1,cdim=20):  
  """
  - generate multiple draws from context sequence
  - calculate distance between reference context and 
  -- target context     = nback / current list
  -- nontarget context  = nback+delta / old list
  return array of distances (i) & (ii)
  """

  # loop vars
  dist2targetL = []
  dist2nontargetL = []
  for itr in range(nitr):
    # sample new context
    C = fast_n_sphere(n_steps=EXP_LEN, dim=cdim, var=cvar, mean=cmean)[0]
    # get index for each condition
    current_t,target_t,lure_t = get_idxs_fn(set_size,delta)
    # target distance
    dist2target_    = distance.cdist(
      C[current_t,:],C[target_t,:], 
      metric=metric)[0]
    # nontarget distance
    dist2nontarget_ = distance.cdist(
      C[current_t,:],C[lure_t,:], 
      metric=metric)[0]
    # collect
    dist2targetL.append(dist2target_)
    dist2nontargetL.append(dist2nontarget_)
    
  dist2target = np.concatenate(dist2targetL)
  dist2nontarget = np.concatenate(dist2nontargetL)
  return dist2target,dist2nontarget
  



""" 
CONTEXT GENERATION
"""

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

