import numpy as np
import torch as tr


class FFWM(tr.nn.Module):
  """ Model used for Sternberg and N-back """
  def __init__(self,indim,hiddim,outdim=2,bias=False,add_noise=False):
    super().__init__()
    self.indim = indim
    self.hiddim = hiddim
    self.add_noise = add_noise
    self.hid1_layer = tr.nn.Linear(indim,indim,bias=bias)
    self.hid2_layer = tr.nn.Linear(indim,hiddim,bias=bias)
    self.out_layer = tr.nn.Linear(hiddim,outdim,bias=bias)
    # self.drop1 = tr.nn.Dropout(p=0.05, inplace=False)
    self.drop2 = tr.nn.Dropout(p=0.05, inplace=False)
    bias_dim = indim
    max_num_bias_modes = 10
    self.embed_bias = tr.nn.Embedding(max_num_bias_modes,bias_dim)
    return None

  def forward(self,inputL,control_bias_int=0):
    """ inputL is list of tensors """
    hid1_in = tr.cat(inputL,-1)
    hid1_act = self.hid1_layer(hid1_in).relu()
    control_bias = self.embed_bias(tr.tensor(control_bias_int))
    hid2_in = hid1_act + control_bias
    if self.add_noise:
      hid2_in = hid2_in + (0.1 ** 0.5) * tr.randn(hid2_in.shape)
    hid2_in = self.drop2(hid2_in)
    hid2_act = self.hid2_layer(hid2_in).relu()
    yhat_t = self.out_layer(hid2_act)
    return yhat_t

def sample_nback_trial(stimset,cdrift,setsize,
  pr_match,pr_stim_lure,pr_context_lure):
  """ 
  one trial of n-back is a 4-tuple:
    s_t,c_t,s_r,c_r
    also returns ytarget
  """
  ntokens,sdim = stimset.shape
  max_context_t = ntokens - setsize
  nsteps,cdim = cdrift.shape
  ## generate train data
  # current stim and context 
  stim_t_idx = np.random.randint(0,ntokens)
  # context_t_idx = np.random.randint(0,max_context_t)
  context_t_idx = np.random.randint(setsize, ntokens)
  stim_t = stimset[stim_t_idx]
  context_t = cdrift[context_t_idx]
  # nback_context_idx = context_t_idx + setsize
  nback_context_idx = context_t_idx - setsize
  # rlo = context_t_idx + 1
  rlo = nback_context_idx - setsize + 1
  # rhi = min(nback_context_idx + 4, ntokens)
  rhi = max(nback_context_idx + setsize - 1, 0)

  ## trial type 
  rand_n = np.random.random()
  ttype_idx = -1
  # both match
  if rand_n < pr_match:
    stim_m = stim_t
    context_m = cdrift[nback_context_idx]
    ytarget = tr.LongTensor([1])
    ttype_idx = 0
  # clure: context match (stim no match) 
  elif rand_n < (pr_match + pr_context_lure):
    idx_stim_m = np.random.choice(np.setdiff1d(range(ntokens), stim_t_idx))
    stim_m = stimset[idx_stim_m]
    context_m = cdrift[nback_context_idx]
    ytarget = tr.LongTensor([0])
    ttype_idx = 1
  # slure: stim match (context no match)
  elif rand_n < (pr_match + pr_context_lure + pr_stim_lure):
    stim_m = stim_t
    # rlo = context_t_idx + 1
    # rhi = min(nback_context_idx + 4, ntokens)
    idx_context_m = np.random.choice(np.setdiff1d(range(rlo, rhi), nback_context_idx))
    # idx_context_m = np.random.choice([nback_context_idx - 1, nback_context_idx + 1])
    context_m = cdrift[idx_context_m]
    ytarget = tr.LongTensor([0])
    ttype_idx = 2
  # neither match
  else:
    idx_stim_m = np.random.choice(np.setdiff1d(range(ntokens), stim_t_idx))
    # rlo = context_t_idx + 1
    # rhi = min(nback_context_idx + 4, ntokens)
    idx_context_m = np.random.choice(np.setdiff1d(range(rlo, rhi), nback_context_idx))
    stim_m = stimset[idx_stim_m]
    context_m = cdrift[idx_context_m]
    ytarget = tr.LongTensor([0])
    ttype_idx = 3
  return stim_t,stim_m,context_t,context_m,ytarget, ttype_idx
  
def sample_nback_trial_strictslure(stimset,cdrift,setsize,
  pr_match,pr_stim_lure,pr_context_lure):
  """ 
  one trial of n-back is a 4-tuple:
    s_t,c_t,s_r,c_r
    also returns ytarget
  """
  min_context_t = setsize
  ntokens,sdim = stimset.shape
  nsteps,cdim = cdrift.shape
  ## generate train data
  # current stim and context 
  stim_t_idx = np.random.randint(0,ntokens)
  context_t_idx = np.random.randint(min_context_t,nsteps)
  stim_t = stimset[stim_t_idx]
  context_t = cdrift[context_t_idx]
  nback_context_idx = context_t_idx - setsize
  ## trial type 
  rand_n = np.random.random()
  # both match
  if rand_n < pr_match:
    stim_m = stim_t
    context_m = cdrift[context_t_idx - setsize]
    ytarget = tr.LongTensor([1])
    ttype_idx = 0
  # clure: context match (stim no match) 
  elif rand_n < (pr_match + pr_context_lure):
    idx_stim_m = np.random.choice(np.setdiff1d(range(ntokens), stim_t_idx))
    stim_m = stimset[idx_stim_m]
    context_m = cdrift[context_t_idx - setsize]
    ytarget = tr.LongTensor([0])
    ttype_idx = 1
  # slure: stim match (context no match)
  elif rand_n < (pr_match + pr_context_lure + pr_stim_lure):
    stim_m = stim_t
    # idx_context_m = np.random.choice(np.setdiff1d(range(nsteps), context_t_idx-setsize))
    # idx_context_m = context_t_idx - setsize + 1
    rlo = context_t_idx + 1
    rhi = min(nback_context_idx + 4, ntokens)
    idx_context_m = np.random.choice(np.setdiff1d(range(rlo, rhi), nback_context_idx))
    context_m = cdrift[idx_context_m]
    ytarget = tr.LongTensor([0])
    ttype_idx = 2
  # neither match
  else:
    idx_stim_m = np.random.choice(np.setdiff1d(range(ntokens), stim_t_idx))
    # idx_context_m = np.random.choice(np.setdiff1d(range(nsteps), context_t_idx-setsize))
    rlo = context_t_idx + 1
    rhi = min(nback_context_idx + 4, ntokens)
    idx_context_m = np.random.choice(np.setdiff1d(range(rlo, rhi), nback_context_idx))
    stim_m = stimset[idx_stim_m]
    context_m = cdrift[idx_context_m]
    ytarget = tr.LongTensor([0])
    ttype_idx = 3
  return stim_t,stim_m,context_t,context_m,ytarget,ttype_idx

def sample_sternberg_trial(stimset,cdrift,setsize,
  pr_match,pr_stim_lure,pr_context_lure):
  """
  training only happens for the last of three trials
    because set-size effect hypothesized to arive out
    of proactive interference with earlier trials
  """
  trial_n = 2
  # task params
  stimset_size,sdim = stimset.shape
  # limit setsize, expand shape to (trial,step,cdim),
  # cdrift = cdrift.reshape(trial_n,-1,100)
  # test context and stim
  # context_t = cdrift[-1,-1,:]
  stim_t_idx = np.random.randint(stimset_size)
  stim_t = stimset[stim_t_idx,:]
  context_t_idx = setsize * trial_n + 1
  context_t = cdrift[context_t_idx]

  ## define positive and negative samples
  stim_pos = stim_t
  context_pos_idx = context_t_idx - np.random.randint(0, setsize)
  context_pos = cdrift[context_pos_idx]
  # negative context, different trial
  context_neg_idx = context_t_idx - np.random.randint(setsize + 1, setsize * 2 + 1)
  context_neg = cdrift[context_neg_idx]
  # stim
  stim_neg_idx = np.random.choice(np.setdiff1d(range(stimset_size), stim_t_idx))
  stim_neg = stimset[stim_neg_idx]
  ## trial type
  randn = np.random.random()
  # both match
  if randn<pr_match:
    # positive trial
    stim_m  = stim_pos
    context_m  = context_pos
    ytarget = tr.LongTensor([1])
    ttype_idx = 0
  # slure: stim match (context no match)
  elif randn<pr_match+pr_stim_lure:
    # stim lure
    stim_m = stim_pos
    context_m = context_neg
    ytarget = tr.LongTensor([0])
    ttype_idx = 2
  # clure: context match (stim no match)
  elif randn<pr_match+pr_stim_lure+pr_context_lure:
    # context lure
    stim_m = stim_neg
    context_m = context_pos
    ytarget = tr.LongTensor([0])
    ttype_idx = 1
  # neither match
  else:
    # negative trial
    stim_m = stim_neg
    context_m = context_neg
    ytarget = tr.LongTensor([0])
    ttype_idx = 3
  return stim_t,context_t,stim_m,context_m,ytarget,ttype_idx


def old_sample_sternberg_trial(stimset,cdrift,setsize,
  pr_match,pr_stim_lure,pr_context_lure):
  """ 
  training only happens for the last of three trials
    because set-size effect hypothesized to arive out
    of proactive interference with earlier trials
  """
  ntrials=2
  # task params
  stimset_size,sdim = stimset.shape
  # limit setsize, expand shape to (trial,step,cdim), 
  cdrift = cdrift[:(setsize+1)*ntrials]
  cdrift = cdrift.reshape(ntrials,-1,100)
  # test context and stim
  context_t = cdrift[-1,-1,:] 
  stim_t_idx = np.random.randint(stimset_size)
  stim_t = stimset[stim_t_idx,:]

  ## define positive and negative samples
  stim_pos = stim_t
  context_pos_idx = context_neg_idx = np.random.randint(0,setsize)
  context_pos = tr.Tensor(cdrift[-1,context_pos_idx,:])  
  # negative context, different trial
  context_neg_idx_trial = np.random.randint(0,ntrials-1)
  context_neg = cdrift[context_neg_idx_trial,context_neg_idx,:]
  # stim
  stim_neg_idx = np.random.choice(np.setdiff1d(range(stimset_size), stim_t_idx))
  stim_neg = stimset[stim_neg_idx,:]
  ## trial type
  randn = np.random.random()
  if randn<pr_match:
    # positive trial
    stim_m  = stim_pos
    context_m  = context_pos
    ytarget = tr.LongTensor([1])
  elif randn<pr_match+pr_stim_lure:
    # stim lure
    stim_m = stim_pos
    context_m = context_neg
    ytarget = tr.LongTensor([0])
  elif randn<pr_match+pr_stim_lure+pr_context_lure:
    # context lure
    stim_m = stim_neg
    context_m = context_pos
    ytarget = tr.LongTensor([0])
  else:
    # negative trial
    stim_m = stim_neg
    context_m = context_neg
    ytarget = tr.LongTensor([0])
  return stim_t,context_t,stim_m,context_m,ytarget

def sample_sternberg_trial_strictslure(
  stimset,cdrift,setsize,pr_match,pr_stim_lure,pr_context_lure):
  """ 
  training only happens for the last of three trials
    because set-size effect hypothesized to arive out
    of proactive interference with earlier trials
  """
  # task params
  stimset_size,sdim = stimset.shape
  nsteps,cdim = cdrift.shape
  ntrials = int(nsteps/(setsize+1))
  # expand shape to (trial,step,cdim), 
  cdrift = cdrift.reshape(ntrials,-1,100)
  # test context and stim
  trial_t = np.random.randint(1,ntrials)
  context_t = cdrift[trial_t,-1,:] 
  stim_t_idx = np.random.randint(stimset_size)
  stim_t = stimset[stim_t_idx,:]
  # define positive samples
  stim_pos = stim_t
  context_pos_idx = context_neg_idx = np.random.randint(0,setsize)
  context_pos = tr.Tensor(cdrift[-1,context_pos_idx,:])  
  # define negative context (i.e. different trial)
  context_neg_trial_idx = np.random.randint(0,trial_t)
  context_neg = cdrift[context_neg_trial_idx,context_neg_idx,:]
  # stim
  stim_neg_idx = np.random.choice(np.setdiff1d(range(stimset_size), stim_t_idx))
  stim_neg = stimset[stim_neg_idx,:]
  ## trial type
  randn = np.random.random()
  if randn<pr_match:
    # positive trial
    stim_m  = stim_pos
    context_m  = context_pos
    ytarget = tr.LongTensor([1])
  elif randn<pr_match+pr_stim_lure:
    # stim lure
    stim_m = stim_pos
    trial_m = trial_t-1
    lure_position = -1 
    context_m = cdrift[trial_m][lure_position-1]
    ytarget = tr.LongTensor([0])
  elif randn<pr_match+pr_stim_lure+pr_context_lure:
    # context lure
    stim_m = stim_neg
    context_m = context_pos
    ytarget = tr.LongTensor([0])
  else:
    # negative trial
    stim_m = stim_neg
    context_m = context_neg
    ytarget = tr.LongTensor([0])
  return stim_t,context_t,stim_m,context_m,ytarget

def sample_stimset(stimset_size,stim_dim):
  stim_bag = np.random.uniform(0,1,[stimset_size,stim_dim])
  return tr.Tensor(stim_bag)

def sample_context_drift(nsteps,noise,cdim=2,delta_M=1):
  """ 
  every element of initial c_t set to anywhere within [0,1]
  steps kept constant
  """
  context_drift = -np.ones([nsteps,cdim])
  c_t = np.random.random(cdim) 
  c_t = np.ones(cdim)
  for step in range(nsteps):
    delta_t = np.random.normal(delta_M,noise,cdim)
    delta_t /= np.linalg.norm(delta_t)
    c_t += delta_t
    context_drift[step] = c_t
  return tr.Tensor(context_drift)


maxsoftmax = lambda x: tr.argmax(tr.softmax(x,-1),-1).squeeze() 

def mov_avg(arr,wind):
  MA = -np.ones(len(arr)-wind)
  for idx in range(len(arr)-wind):
    MA[idx] = arr[idx:idx+wind].mean()
  return MA

