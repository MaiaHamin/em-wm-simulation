import numpy as np
import torch as tr

def single_sternberg_comparison(stimset,cdrift,setsize,
  pr_match,pr_stim_lure,pr_context_lure):
  # task params
  stimset_size,sdim = stimset.shape

  # set current stim and context
  stim_t_idx = np.random.randint(stimset_size)
  stim_t = stimset[stim_t_idx,:]
  context_t_idx = setsize * 2 + 1 # the current trace is always the test probe from the second list
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

  # trial type
  ttype_randn = np.random.random()

  # both match
  if ttype_randn<pr_match:
    # positive trial
    stim_m  = stim_pos
    context_m  = context_pos
    ytarget = tr.LongTensor([1])
    ttype_idx = 0
  # slure: stim match (context no match)
  elif ttype_randn<pr_match+pr_stim_lure:
    # stim lure
    stim_m = stim_pos
    context_m = context_neg
    ytarget = tr.LongTensor([0])
    ttype_idx = 2
  # clure: context match (stim no match)
  elif ttype_randn<pr_match+pr_stim_lure+pr_context_lure:
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

def single_nback_comparison(stimset,cdrift,setsize,
  pr_match,pr_stim_lure,pr_context_lure):
  """ 
  Returns a pair of a stimulus-context traces representing a currently-active stimulus and context & a single trace
  retrieved from memory. Generates this pair based on the four comparison types laid out in the paper:
    both-match, stim lure, context lure, neither match.
  output is a 4-tuple (s_t,c_t,s_r,c_r), as well as ytarget and ttype_code which encodes the trial type for error reporting.
  """
  ntokens,sdim = stimset.shape
  min_context_t = setsize

  # set current stim and context
  stim_t_idx = np.random.randint(0,ntokens)
  context_t_idx = np.random.randint(min_context_t, ntokens)
  stim_t = stimset[stim_t_idx]
  context_t = cdrift[context_t_idx]

  ttype_randn = np.random.random()  # randomly-selected trial type
  ttype_code = -1 # code used to record trial type for analysis

  if ttype_randn < pr_match:
    stim_m, context_m = nback_both_match(stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens)
    ytarget = tr.LongTensor([1])
    ttype_code = 0

  elif ttype_randn < (pr_match + pr_context_lure):
    stim_m, context_m = nback_ctxt_lure(stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens)
    ytarget = tr.LongTensor([0])
    ttype_code = 1

  elif ttype_randn < (pr_match + pr_context_lure + pr_stim_lure):
    stim_m, context_m = nback_stim_lure(stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens)
    ytarget = tr.LongTensor([0])
    ttype_code = 2

  else:
    stim_m, context_m = nback_neither_match(stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens)
    ytarget = tr.LongTensor([0])
    ttype_code = 3

  return stim_t,stim_m,context_t,context_m,ytarget, ttype_code

def sample_nback_retrieval_set_for_trial_sequence(stimset, cdrift, setsize,
                                          pr_match, pr_stim_lure, pr_context_lure):
    """
    Simulates the set of traces in EM corresponding to a possible n-back sequence for a particular trial type,
    returning an list of tuples (with the current trial is at index 0)
    of stimulus and context information, as well as the true answer ytarget and a code for the trial type.
    """
    n_traces = 10
    trace_seq = [None] * n_traces
    ntokens,sdim = stimset.shape
    min_context_t = setsize

    ## current stim and context
    stim_t_idx = np.random.randint(0,ntokens)
    context_t_idx = np.random.randint(min_context_t, ntokens)
    stim_t = stimset[stim_t_idx]
    context_t = cdrift[context_t_idx]
    trace_seq[0] = (stim_t, context_t) # the 0th index is the current stim, context

    ttype_rand_n = np.random.random() # random number for determining the trial type
    ttype_code = -1  # code for the trial type that will be returned - 0 if a control, and 1 if a lure

    # control target
    if ttype_rand_n < pr_match:
        # One trace in the n-back position with a matching stimulus
        trace_seq[setsize] = nback_both_match(stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens)
        # All other traces in the sequence are from non-nback positions and have non-matching stimuli
        for i in range(1, n_traces):
          if i != setsize:
              trace_seq[i] = nback_neither_match(stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens)
        ytarget = tr.LongTensor([1])
        ttype_code = 0

    # control foil
    elif ttype_rand_n < (pr_match + pr_context_lure):
        # One trace in the n-back position with a non-matching stimulus
        trace_seq[setsize] = nback_ctxt_lure(stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens)
        # All other traces in the sequence have non-matching stimuli
        for i in range(1, n_traces):
            if i != setsize:
                trace_seq[i] = nback_neither_match(stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens)
        ytarget = tr.LongTensor([0])
        ttype_code = 0

    # lure target
    elif ttype_rand_n < (pr_match + pr_context_lure + pr_stim_lure):
        # One trace in the n-back position with a matching stimulus
        trace_seq[setsize] = nback_both_match(stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens)
        # One trace in the non n-back position has a non-matching stimulus
        trace_seq[setsize - 1] = nback_stim_lure(stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens)
        # All other traces have non-matching stimuli
        for i in range(1, n_traces):
            if i != setsize and i != setsize - 1:
                trace_seq[i] = nback_neither_match(stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens)
        ytarget = tr.LongTensor([1])
        ttype_code = 1

    # lure foil
    else:
      # The trace in the n-back position has a non-matching stimulus
      trace_seq[1] = nback_ctxt_lure(stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens)
      # The trace one-after the n-back position (the lure position) has a matching stimulus
      trace_seq[2] = nback_stim_lure(stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens)
      # All other traces in the sequence other than the n-back position have non-matching stimuli
      for i in range(3, n_traces):
        trace_seq[i] = nback_neither_match(stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens)
      ytarget = tr.LongTensor([0])
      ttype_code = 1

    return trace_seq, ytarget, ttype_code

# return a both match trace -- the stimuli match, and the context is the n-back context
def nback_both_match(stim_set, cdrift, stim_t_idx, context_t_idx, setsize, ntokens):
    stim_m = stim_set[stim_t_idx]
    context_m = cdrift[context_t_idx - setsize]
    return (stim_m, context_m)

# return a stim lure trace -- the stimuli match, but the context is not the n-back context
def nback_stim_lure(stim_set, cdrift, stim_t_idx, context_t_idx, setsize, ntokens):
    stim_m = stim_set[stim_t_idx]
    context_m = get_lure_context(cdrift, context_t_idx, ntokens, setsize)
    return (stim_m, context_m)

# return a context lure trace -- the stimuli don't match, but the context is the n-back context
def nback_ctxt_lure(stim_set, cdrift, stim_t_idx, context_t_idx, setsize, ntokens):
    idx_stim_m = np.random.choice(np.setdiff1d(range(ntokens), stim_t_idx))
    stim_m = stim_set[idx_stim_m]
    context_m = cdrift[context_t_idx - setsize]
    return (stim_m, context_m)

# return a neither match trace -- the stimuli don't match, and the context is not the n-back context.
# optionally, for the EM simulations, this can probabilistically return a matching stimulus and a long-past context
# (s.t. the trace isn't a proper lure, but simulates the repeating-stimuli dynamics of the task).
def nback_neither_match(stim_set, cdrift, stim_t_idx, context_t_idx, setsize, ntokens, pr_prewindow_match = 0.0):
    if np.random.uniform() > pr_prewindow_match or ntokens - context_t_idx < 6:
        idx_stim_m = np.random.choice(np.setdiff1d(range(ntokens), stim_t_idx))
        stim_m = stim_set[idx_stim_m]
        context_m = get_lure_context(cdrift, context_t_idx, ntokens, setsize)
        return stim_m, context_m
    else:
        return nback_distant_slure(stim_set, cdrift, stim_t_idx, context_t_idx, setsize, ntokens)

# A trial type where the stimulus DOES match, but the context is so far from the target context that it isn't a lure
def nback_distant_slure(stim_set, cdrift, stim_t_idx, context_t_idx, setsize, ntokens):
    idx_stim_m = np.random.choice(np.setdiff1d(range(ntokens), stim_t_idx))
    nback_context_idx = context_t_idx + setsize
    rlo = max(0, nback_context_idx - 6)
    rhi = min(nback_context_idx - 2, ntokens)
    idx_context_m = np.random.choice(range(rlo, rhi))
    stim_m = stim_set[idx_stim_m]
    context_m = cdrift[idx_context_m]
    return stim_m, context_m

def get_lure_context(cdrift, context_t_idx, ntokens, setsize):
    try:
        nback_context_idx = context_t_idx - setsize
        rlo = max(0, nback_context_idx - 2)
        rhi = min(nback_context_idx + setsize, ntokens)
        idx_context_m = np.random.choice(np.setdiff1d(range(rlo, rhi), nback_context_idx))
        context_m = cdrift[idx_context_m]
        return context_m
    except:
        print("uh oh")
        print(nback_context_idx)
        print(rlo)
        print(rhi)
        return
