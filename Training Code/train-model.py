import os
import sys
from datetime import datetime as dt
from glob import glob as glob
import torch as tr
import numpy as np
import itertools
from matplotlib import pyplot as plt
from ModelsFFWM import *
sys.path.append("/Users/maia/Projects/thesis-code/drift_fn_models")
from drift_fns import context_integration_diffusion_longwalk, \
  andre_drift, unnormed_context_integration_diffusion_multistep, \
  simple_scaled_context_integration_diffusion_multistep, iid_noise, \
  path_integration, n_sphere, fast_n_sphere, multiscale_n_sphere
from generate_stimset import gen_stims

from sequence_dprime import calculate_dprime
from sampling_dprime import sampling_dprime

def run_model(net,taskL,cfun,training,neps_per_task,stim_ntokens,sdim,n_context_steps,stim_v,verb=True):
  if training: net.train()
  else: net.eval()
  lossop = tr.nn.CrossEntropyLoss()
  optiop = tr.optim.Adam(net.parameters(), lr=0.001)
  n_ttypes = 4
  score = -np.ones([len(taskL), neps_per_task])
  ttype = -np.ones([len(taskL), neps_per_task])
  for ep in range(neps_per_task):
    stimset = gen_stims(stim_v)  # tr.load(stimset_name + ".tr")
    if verb and ep % (neps_per_task / 5) == 0:
      print(ep / neps_per_task)
    # resample stim and context on each ep
    # cdrift = sample_context_drift(nsteps=num_context_steps,noise=context_noise)
    cdrift = cfun(n_steps=n_context_steps, idx=ep)
    # simple_scaled_context_integration_diffusion_multistep(stims=[], n_steps=num_context_steps, stim_d=cdim)
    # andre_drift(stims=[], n_steps=num_context_steps) # #
    cdrift = tr.Tensor(cdrift)
    # interleave train on every task per epoch
    for task_idx,(control_int,sample_trial_fn,setsize) in enumerate(taskL): 
      # generate trial sample
      out = sample_trial_fn(stimset,cdrift,setsize)
      stim_t,stim_m,context_t,context_m,ytarget, ttype_idx = out
      # forward prop
      inputL = [stim_t,stim_m,context_t,context_m]
      yhat = net(inputL,control_bias_int=control_int)
      # eval
      score[task_idx, ep] = maxsoftmax(yhat)==ytarget

      ttype[task_idx, ep] = ttype_idx
      # backprop
      if training:
        eploss = lossop(yhat.unsqueeze(0), ytarget)
        optiop.zero_grad()
        eploss.backward(retain_graph=True)
        optiop.step()
  return score, ttype


def train_model_intervals(net,taskL,long_cfun,cfun,training,neps_per_task,stim_ntokens,sdim,n_context_steps,model_name,stim_v,verb=True):
  if training: net.train()
  else: net.eval()
  lossop = tr.nn.CrossEntropyLoss()
  optiop = tr.optim.Adam(net.parameters(), lr=0.001)
  n_ttypes = 4
  score = -np.ones([len(taskL), neps_per_task])
  ttype = -np.ones([len(taskL), neps_per_task])
  # stimset = tr.eye(stim_ntokens,sdim)
  for ep in range(neps_per_task):
    if verb and ep % (neps_per_task / 5) == 0:
      print(ep / neps_per_task)
    stimset = gen_stims(stim_v)  # tr.load(stimset_name + ".tr")
    # resample stim and context on each ep
    # cdrift = sample_context_drift(nsteps=num_context_steps,noise=context_noise)
    cdrift = long_cfun(n_steps=n_context_steps, idx=ep)
    # simple_scaled_context_integration_diffusion_multistep(stims=[], n_steps=num_context_steps, stim_d=cdim)
    # andre_drift(stims=[], n_steps=num_context_steps) # #
    cdrift = tr.Tensor(cdrift)
    # interleave train on every task per epoch
    for task_idx,(control_int,sample_trial_fn,setsize) in enumerate(taskL):
      # generate trial sample
      out = sample_trial_fn(stimset,cdrift,setsize)
      stim_t,stim_m,context_t,context_m,ytarget, ttype_idx = out
      # forward prop
      inputL = [stim_t,stim_m,context_t,context_m]
      yhat = net(inputL,control_bias_int=control_int)
      # eval
      score[task_idx, ep] = maxsoftmax(yhat)==ytarget

      ttype[task_idx, ep] = ttype_idx
      # backprop
      if training:
        eploss = lossop(yhat.unsqueeze(0), ytarget)
        optiop.zero_grad()
        eploss.backward(retain_graph=True)
        optiop.step()
    # if ep % 25000 == 0:
      # sampling_dprime(net, model_name, cfun, mean, var, beta, dim, fname='../Figures/interval_training/' + str(ep) + "_")
      # np.save('interval_nets/trsc-' + model_name + str(ep), score)
      # tr.save(net.state_dict(), 'interval_nets/trsc-' + model_name + str(ep) + ".pt")

  return score, ttype


def long_walk(long_context, idx=0, n_steps=20):
  n_idx = idx % int(len(long_context) / n_steps - 1)
  ctxt = long_context[n_idx * n_steps: (n_idx + 1) * n_steps]
  if len(ctxt) < n_steps:
    raise ValueError(
      "too short with idx " + str(idx) + " using n_idx " + str(n_idx) + " and len " + str(len(long_context)))
  return ctxt

# vars
def run_all(seed, ctxt_fn, mean, var, beta, dim, neps):
  np.random.seed(seed)
  tr.random.manual_seed(seed)

  # task params
  task = "nback" # "stern"
  stim_ntokens = 20
  num_context_steps = 20

  ## net params
  sdim = stim_ntokens
  indim = 2 * (dim + sdim)
  hiddim = dim * 4

  # train / ev params
  stimset_name = "stimset-v=075"
  pr_match, pr_slure, pr_clure = 0.4, 0.2, 0.2
  # model_name = 'ffwm-%s-ss49strictlure-tr_noise_%i-seed_%i'%(task,noise_tr*10,seed)
  model_name = 'ffwm-ttype%s_probs%i%i%i--neps%i-seed%i' % (
  task, pr_match * 10, pr_slure * 10, pr_clure * 10, neps, seed)
  long_context, fn_name = ctxt_fn(var=var, mean=mean, beta=beta, dim=dim, n_steps=neps)
  # simple_scaled_context_integration_diffusion_multistep(stims=[])
  # andre_drift(stims=[])
  model_name += "_" + fn_name + "_" + stimset_name + "_swap-vars_"
  model_name = model_name.replace(".", "")
  param_string = 'm=%.2f, v=%.2f, b=%.2f, dim=%i, neps=%i' % (mean, var, beta, dim, neps)
  print(param_string)
  ## init task and net

  # task sampling
  # sample_task(pr_match,pr_slure,pr_clure)
  if task == 'nback':
    ssizeL = [2, 3] # [1, 3]
    sample_fn = lambda match,slure,clure: lambda S,C,N: sample_nback_trial(S,C,N,
                  pr_match=match,pr_stim_lure=slure,pr_context_lure=clure
                  )
  elif task == 'stern':
    ssizeL = [4,9]
    sample_fn = lambda match,slure,clure: lambda S,C,N: sample_sternberg_trial(S,C,N,
                  pr_match=match,pr_stim_lure=slure,pr_context_lure=clure
                  )



  taskintL = [0,1]
  taskL_tr = [
    [taskintL[0],sample_fn(pr_match,pr_slure,pr_clure),ssizeL[0]],
    [taskintL[1],sample_fn(pr_match,pr_slure,pr_clure),ssizeL[1]],
  ]

  # init net
  net = FFWM(indim,hiddim)

  ## train
  # score_tr, ttype = run_model(net,taskL_tr,
  #                     cfun = lambda idx, n_steps: long_walk(long_context, idx, n_steps),
  #                     # cfun=lambda idx, n_steps: ctxt_fn(var=var, mean=mean, beta=beta, dim=dim, n_steps=n_steps)[0],
  #                     stim_ntokens=stim_ntokens,
  #                     sdim=sdim,
  #                     n_context_steps=num_context_steps,
  #                      training=True,
  #                      neps_per_task=neps)

  score_tr, ttype = train_model_intervals(net, taskL_tr,
                              long_cfun=lambda idx, n_steps: long_walk(long_context, idx, n_steps),
                              cfun=ctxt_fn,
                              # cfun=lambda idx, n_steps: ctxt_fn(var=var, mean=mean, beta=beta, dim=dim, n_steps=n_steps)[0],
                              stim_ntokens=stim_ntokens,
                              sdim=sdim,
                              n_context_steps=num_context_steps,
                              training=True,
                              neps_per_task=neps,
                              model_name=model_name,
                              stim_v=0.25)

  np.save('2and3back/trsc-'+model_name,score_tr)
  tr.save(net.state_dict(), '2and3back/trsc-'+model_name + ".pt")

  ## eval
  print('eval')
  neps_ev = 1000
  plot_train_acc(score_tr, ttype, fn_name, model_name, taskintL, ssizeL, net,
                 sample_fn, neps_ev, ctxt_fn, long_context,
                 stim_ntokens, sdim, num_context_steps)

  ## plot train acc

def plot_train_acc(score_tr, ttype, fn_name, model_name, taskintL, ssizeL, net,
                 sample_fn, neps_ev, ctxt_fn, long_context,
                 stim_ntokens, sdim, num_context_steps):
  c = ["orange", "blue"]
  m = ["o", "x", "v", "^"]

  tasks = ["2-back", "3-back"]
  colors = [["#edc174", "#d4982f", "#c27e08", "#a1690a"], ["#8dcbf7", "#61b3ed", "#2a86c7", "#0762a3"]]
  labels = ["match", "slure", "clure", "nomatch"]
  n_intervals = 1000
  for i in range(len(score_tr)):
    task_score = score_tr[i]
    task_color = colors[i]
    task_trialtypes = ttype[i]
    for tt in range(4):
      filt_inds = task_trialtypes == tt
      ep_ttype = np.extract(filt_inds, task_score)
      ep_ttype = ep_ttype[:-(len(ep_ttype)%n_intervals)]
      ac = ep_ttype.reshape(-1, n_intervals).mean(1)
      lt = tasks[i] + " " + labels[tt]
      plt.plot(ac, color=task_color[tt], label=lt)
  plt.legend(loc="best")
  plt.ylim(0, 1)
  plt.ylabel('Train accuracy, ' + fn_name)
  plt.savefig('../Figures/2and3back/trac_%s'%(model_name))
  plt.close('all')

  # fig,axarr = plt.subplots(2,3,figsize=(20,8));axarr=axarr.reshape(-1)
  # for nidx,noise_ev in enumerate([0,0.2,0.4,0.6,0.8,1.0]):
  # ax = axarr[nidx]
  for tidx, (task_int, ssize) in enumerate(zip(taskintL, ssizeL)):
    plt.title('Accuracy by trial type, ' + fn_name)
    evac = eval_byttype(net, sample_fn, ssize, task_int, neps_ev,
                        long_context, stim_ntokens, sdim, num_context_steps)
    np.save('exp-onehot_stim/evac-' + model_name + '-ssize_%i' % (ssize), evac)
    plt.bar(np.arange(4) + (.45 * tidx), evac, width=.45)
    # plt.ylim(0, 1)
  plt.xticks(range(4), ['match', 'nomatch', 'slure', 'clure'])
  plt.savefig('../Figures/2and3back/ttype_eval_%s' % (model_name))
  plt.close('all')

  sampling_dprime(net, model_name, ctxt_fn, mean, var, beta, dim, "../Figures/2and3back/")


def eval_byttype(net, sample_fn, ssize, task_int, neps, long_context,
                 stim_ntokens, sdim, num_context_steps):
    """ eval on given task for separate trial types
    returns evac on (match,nomatch,slure,clure)
    """
    taskL_ev = [
      [task_int, sample_fn(1, 0, 0), ssize],
      [task_int, sample_fn(0, 0, 0), ssize],
      [task_int, sample_fn(0, 1, 0), ssize],
      [task_int, sample_fn(0, 0, 1), ssize]
    ]
    evsc, ttype = run_model(
      net, taskL_ev,
      cfun=lambda idx, n_steps: long_walk(long_context, idx, n_steps),
      # cfun=lambda idx, n_steps: ctxt_fn(var=var, mean=mean, beta=beta, dim=dim, n_steps=n_steps)[0],
      stim_ntokens=stim_ntokens,
      sdim=sdim,
      n_context_steps=num_context_steps,
      training=False,
      neps_per_task=neps,
      stim_v=0.25,
      verb=False
    )
    evac = evsc.mean(1)
    return evac





walk_drift = iid_noise, [(0.1, 0.0, 0.0), (0.1, 0.1, 0.0), (0.1, 0.2, 0.0), (0.5, 0.5, 0.0), (0.5, 0.25, 0.0), (0.5, 0.75, 0.0)]
walk_diffusion = iid_noise, [(0.0, 0.1, 0.0), (0.0, 0.25, 0.0), (0.0, 0.5, 0.0), (0.0, 1.0, 0.0)]

sphere_drift = fast_n_sphere, [(0.25, 0.0, 0.0), (0.25, 0.05, 0.0), (0.25, 0.075, 0.0), (0.4, 0.075, 0.0)]
sphere_diffusion = n_sphere, [(0.0, 0.1, 0.0), (0.0, 0.25, 0.0), (0.0, 0.5, 0.0), (0.0, 0.75, 0.0)]

# path_drift = path_integration, [(1.0, 1.0, 0.5), (1.0, 0.5, 0.5), (0.5, 0.5, 0.5)]
path_diffusion = path_integration, [(0.0, 1.0, 0.1), (0.0, 1.0, 0.3), (0.0, 0.5, 0.3), (0.0, 0.5, 0.3)]

andre_driftfn = andre_drift, [(1.0, 0.5, 0.0), (1.0, 1.0, 0.0), (1.0, 2.0, 0.0)]
andre_diffusion = andre_drift, [(0.0, 0.5, 0.0), (0.0, 1.0, 0.0), (0.0, 2.0, 0.0)]

andre_driftfn = andre_drift, [(0.1, 0.1, 0.0), (0.1, 0.2, 0.0), (0.25, 0.25, 0.0), (0.25, 0.3, 0.0)]
andre_diffusion = andre_drift, [(0.0, 0.1, 0.0), (0.0, 0.2, 0.0), (0.0, 0.5, 0.0)]


multiscale_sphere_drift = multiscale_n_sphere, [(0.75, 0.0, 0.0), (0.5, 0.0, 0.0), (0.25, 0.0, 0.0)]

functions = [sphere_drift]

for seed in [1, 2]:
  for function in functions:
    fn = function[0]
    params = function[1]
    for dim in [25]:
      for (mean, var, beta) in params:
          run_all(seed=seed, ctxt_fn=fn, mean=mean, dim=dim, var=var, beta=beta, neps=400000)
