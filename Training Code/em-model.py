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
    simple_scaled_context_integration_diffusion_multistep, \
    iid_noise, path_integration, n_sphere, fast_n_sphere
from scipy.stats import norm
import glob
from counting_model import FFNN_mod
from sklearn.metrics.pairwise import cosine_similarity
from generate_stimset import gen_stims
from matplotlib.colors import to_rgba

## net params
stim_ntokens = 20
num_context_steps = 20
cdim = 25
sdim = stim_ntokens
indim = 2*(cdim+sdim)
hiddim = cdim * 4
add_noise = False
flip_bits = True

n_tasks = 2
task = "nback"
if task == 'nback':
  ssizeL = [2, 3] # [1, 3]
  sample_fn = lambda match,slure,clure: lambda S,C,N: sample_nback_trial(S,C,N,
                pr_match=match,pr_stim_lure=slure,pr_context_lure=clure
                )
elif task == 'stern':
  ssizeL = [4,9]
  sample_fn = lambda match,slure,clure: lambda S,C,N: sample_sternberg_trial_strictslure(S,C,N,
                pr_match=match,pr_stim_lure=slure,pr_context_lure=clure
                )

def cos_sim(a, b):
#     return np.dot(a, b) / (np.sqrt(np.dot(a,a)) * np.sqrt(np.dot(b, b)))
    return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))

def bit_flip_noise(ctxt):
    for ind in range(len(ctxt)):
        if np.random.uniform(0, 1) < 0.0:
            if (ctxt[ind] > 0.5):
                ctxt[ind] = ctxt[ind] - 1
            else:
                ctxt[ind] = ctxt[ind] + 1
    return ctxt

def sample_nback_trial(stimset,cdrift,setsize,
  pr_match,pr_stim_lure,pr_context_lure):
  """
  one trial sequence for n-back is a list of tuples of len setsize+1, in reverse order of presentation:
    [(s_t,c_t),(s_t-1,c_t-1), ...]
    also returns ytarget and trial type
  """
  stimseq = [None] * (10)
  ntokens,sdim = stimset.shape
  max_context_t = ntokens - setsize
  nsteps,cdim = cdrift.shape
  ## generate train data
  # current stim and context
  stim_t_idx = np.random.randint(0,ntokens)
  context_t_idx = np.random.randint(max_context_t)
  stim_t = stimset[stim_t_idx]
  context_t = cdrift[context_t_idx]
  stimseq[0] = (stim_t, context_t)
  ## trial type
  rand_n = np.random.random()
  # control target
  if rand_n < pr_match:
    stimseq[1] = both_match(stimset, cdrift, stim_t_idx, context_t_idx, setsize, nsteps, ntokens)
    for i in range(2, 10):
        stimseq[i] = neither_match(stimset, cdrift, stim_t_idx, context_t_idx, setsize, nsteps, ntokens)
    ytarget = tr.LongTensor([1])
    ttype = 0 # "control target"
  # control foil
  elif rand_n < (pr_match + pr_context_lure):
    stimseq[1] = ctxt_lure(stimset, cdrift, stim_t_idx, context_t_idx, setsize, nsteps, ntokens)
    for i in range(2, 10):
        stimseq[i] = neither_match(stimset, cdrift, stim_t_idx, context_t_idx, setsize, nsteps, ntokens)
    ytarget = tr.LongTensor([0])
    ttype = 0 # "control foil"
  # lure target
  elif rand_n < (pr_match + pr_context_lure + pr_stim_lure):
    stimseq[1] = both_match(stimset, cdrift, stim_t_idx, context_t_idx, setsize, nsteps, ntokens)
    stimseq[2] = stim_lure(stimset, cdrift, stim_t_idx, context_t_idx, setsize, nsteps, ntokens)
    for i in range(3, 10):
        stimseq[i] = neither_match(stimset, cdrift, stim_t_idx, context_t_idx, setsize, nsteps, ntokens)
    ytarget = tr.LongTensor([1])
    ttype = 1 # "lure target"
  # lure foil
  else:
    stimseq[1] = ctxt_lure(stimset, cdrift, stim_t_idx, context_t_idx, setsize, nsteps, ntokens)
    stimseq[2] = stim_lure(stimset, cdrift, stim_t_idx, context_t_idx, setsize, nsteps, ntokens)
    for i in range(3, 10):
        stimseq[i] = neither_match(stimset, cdrift, stim_t_idx, context_t_idx, setsize, nsteps, ntokens)
    ytarget = tr.LongTensor([0])
    ttype = 1 # "lure foil"
  return stimseq,ytarget, ttype

def stim_lure(stim_set, cdrift, stim_t_idx, context_t_idx, setsize, nsteps, ntokens):
    stim_m = stim_set[stim_t_idx]
    nback_context_idx = context_t_idx + setsize
    rlo = context_t_idx + 1
    rhi = min(nback_context_idx + 4, ntokens)
    idx_context_m = np.random.choice(np.setdiff1d(range(rlo, rhi), nback_context_idx))
    context_m = cdrift[idx_context_m]
    if flip_bits:
        context_m = bit_flip_noise(context_m)
    return (stim_m, context_m)

def both_match(stim_set, cdrift, stim_t_idx, context_t_idx, setsize, nsteps, ntokens):
    stim_m = stim_set[stim_t_idx]
    context_m = cdrift[context_t_idx + setsize]
    if flip_bits:
        context_m = bit_flip_noise(context_m)
    return (stim_m, context_m)

def ctxt_lure(stim_set, cdrift, stim_t_idx, context_t_idx, setsize, nsteps, ntokens):
    idx_stim_m = np.random.choice(np.setdiff1d(range(ntokens), stim_t_idx))
    stim_m = stim_set[idx_stim_m]
    context_m = cdrift[context_t_idx + setsize]
    if flip_bits:
        context_m = bit_flip_noise(context_m)
    return (stim_m, context_m)

def neither_match(stim_set, cdrift, stim_t_idx, context_t_idx, setsize, nsteps, ntokens):
    pr_prewindow_slures = 0.1
    if np.random.uniform() > pr_prewindow_slures or ntokens - context_t_idx < 6:
        idx_stim_m = np.random.choice(np.setdiff1d(range(ntokens), stim_t_idx))
        nback_context_idx = context_t_idx + setsize
        rlo = context_t_idx + 1
        rhi = min(nback_context_idx + 4, ntokens)
        idx_context_m = np.random.choice(np.setdiff1d(range(rlo, rhi), nback_context_idx))
        stim_m = stim_set[idx_stim_m]
        context_m = cdrift[idx_context_m]
        if flip_bits:
            context_m = bit_flip_noise(context_m)
        return stim_m, context_m
    else:
        return pre_window_slure(stim_set, cdrift, stim_t_idx, context_t_idx, setsize, nsteps, ntokens)

def pre_window_slure(stim_set, cdrift, stim_t_idx, context_t_idx, setsize, nsteps, ntokens):
    idx_stim_m = np.random.choice(np.setdiff1d(range(ntokens), stim_t_idx))
    nback_context_idx = context_t_idx + setsize
    rlo = nback_context_idx + 1
    rhi = min(nback_context_idx + 10, ntokens)
    idx_context_m = np.random.choice(range(rlo, rhi))
    stim_m = stim_set[idx_stim_m]
    context_m = cdrift[idx_context_m]
    if flip_bits:
        context_m = bit_flip_noise(context_m)
    return stim_m, context_m

def run_model(net,taskL,cfun,training,neps_per_task,stim_ntokens,sdim,n_context_steps,sweight,cweight,nret,hrate,stim_v,sim_thresh,verb=True):
  net.eval()
  score = -np.ones([len(taskL), neps_per_task, 2])
  ttype = -np.ones([len(taskL), neps_per_task])
  # stimset = tr.eye(stim_ntokens,sdim)
  stimset = gen_stims(stim_v)
  for ep in range(neps_per_task):
    if verb and ep % (neps_per_task / 5) == 0:
      print(ep / neps_per_task)
    # resample stim and context on each ep
    # cdrift = sample_context_drift(nsteps=num_context_steps,noise=context_noise)
    cdrift = cfun(n_steps=n_context_steps, idx=ep)
    cdrift = tr.Tensor(cdrift)
    # interleave train on every task per epoch
    for task_idx,(control_int,sample_trial_fn,setsize) in enumerate(taskL):
      # generate trial sample
      stimseq,ytarget, ttype_idx = sample_trial_fn(stimset,cdrift,setsize)
      target = stimseq[0]
      sim_scores = []
      for memory in stimseq[1:]:
          cs_stim = cos_sim(target[0], memory[0])
          cs_ctxt = cos_sim(target[1], memory[1])
          sim = sweight * cs_stim + cweight * cs_ctxt
          sim_scores.append(sim)
      pred = 0
      for i in range(len(sim_scores)):
          max_sim_idx = np.argmax(np.array(sim_scores))
          if (sim_scores[max_sim_idx] < sim_thresh):
              break
          # forward prop
          memory = stimseq[max_sim_idx + 1]
          # print("ttype idx:")
          # print(ttype_idx)
          # print("sim scores:")
          # print(sim_scores)
          # print("picked:")
          # print(max_sim_idx)
          inputL = [target[0],memory[0],target[1],memory[1]]
          yhat = net(inputL,control_bias_int=control_int)
          pred = maxsoftmax(yhat).item()
          if pred == 1:
              break
          if np.random.uniform() > (hrate): #** (i + 1)):
              break
          else:
              sim_scores[max_sim_idx] = 0
      # print("predicted:")
      # print(maxsoftmax(yhat).item())
      # eval
      score[task_idx, ep, 0] = pred
      score[task_idx, ep, 1] = ytarget.item()
      ttype[task_idx, ep] = ttype_idx
      # print("=======")
  return score, ttype

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def paper_dprime(hrate, farate):
    hrate = clamp(hrate, 0.01, 0.99)
    farate = clamp(farate, 0.01, 0.99)
    dl = np.log(hrate * (1 - farate) / ((1 - hrate) * farate))
    c = 0.5 * np.log((1 - hrate) * (1 - farate) / (hrate * farate))
    return dl, c

def long_walk(long_context, idx=0, n_steps=20):
  n_idx = idx % int(len(long_context) / n_steps - 1)
  ctxt = long_context[n_idx * n_steps: (n_idx + 1) * n_steps]
  if len(ctxt) < n_steps:
    raise ValueError(
      "too short with idx " + str(idx) + " using n_idx " + str(n_idx) + " and len " + str(len(long_context)))
  return ctxt

path_base = "/Users/maia/Projects/thesis-code/task_models/param_search/"
# paths = []
# # for seed in seeds:
# #     subpaths = glob.glob(path_base + "*seed" + str(seed) + "*" + ".pt")
# #     for sp in subpaths:
# #         paths.append(sp)

seq_types = ["control", "lure"]
trial_types = ["target", "foil"]

# csv_writer = csv.writer(open('sequence_dprime.csv', mode='w'))
# csv_writer.writerow(["dim", "beta", "var",
#                      "d' 2back control", "bias 2back control",
#                      "d' 2back lure", "bias 2back lure",
#                      "d' 4back control", "bias 4back control"
#                      "d' 4back lure", "bias 4back lure"])

paths = glob.glob(path_base + "*path_integration_diffusion_d=10_m=00_v=10*.pt")
ctxt_fn = n_sphere # iid_noise # andre_drift # unnormed_context_integration_diffusion_multistep # simple_scaled_context_integration_diffusion_multistep
has_beta = True
# paths = [path_base + "trsc-ffwm-ttypenback_probs222--neps200000-seed1_random_walk_drift_d=10_m=10_v=00.pt"]

def parse_params(path):
    v_ind = str(path).find("v=")

    dim_ind = str(path).find("d=")
    multi_ind = str(path).find("multi")
    m_ind = str(path).find("m=")

    if has_beta:
        # b_ind = str(path).find("beta=")
        # b = {"beta=02": 0.2, "beta=04": 0.4, "beta=05": 0.5, "beta=06": 0.6}[(path[b_ind: multi_ind - 1])]
        b_ind = str(path).find("b=")
        b = {"b=00": 0.0, "b=025": 0.25, "b=04": 0.4, "b=05": 0.5, "b=06": 0.6, "b=075": 0.75}[
            (path[b_ind: str(path).find("pt") - 1])]
    else:
        b_ind = str(path).find("pt")
        b = 0.0

    v = {"v=005": 0.05, "v=025": 0.25, "v=05": 0.5, "v=01": 0.1, "v=00": 0.0, "v=10": 1.0, "v=20": 2.0, "v=075": 0.75}[
        path[v_ind: b_ind - 1]]
    m = {"m=00": 0.00, "m=025": 0.25, "m=05": 0.5, "m=01": 0.1, "m=10": 1.0}[path[m_ind: v_ind - 1]]
    # cdim = int(path[dim_ind + 2 : m_ind - 1])
    cdim = 10

    indim = 2 * (cdim + sdim)
    hiddim = 4 * cdim
    net = FFWM(indim, hiddim, add_noise=False)
    net.load_state_dict(tr.load(str(path)))

    return net, v, m


def get_perf(net, model_name, ctxt_fn, m, v, b, cdim, sweight, cweight, nret, hrate, stim_v, sim_thresh):
    taskL = [0, 1]
    taskL_ev = [
        [taskL[0], sample_fn(0.25, 0.25, 0.25), ssizeL[0]],
        [taskL[1], sample_fn(0.25, 0.25, 0.25), ssizeL[1]]
    ]

    long_context, fn_name = ctxt_fn(var=v, mean=m, beta=b, dim=cdim, n_steps=100000)
    # _, fn_name = ctxt_fn(var=v, mean=m, beta=b, dim=cdim, n_steps=1)

    neps = 5000
    evsc, ttype = run_model(
        net, taskL_ev,
        cfun=lambda idx, n_steps: long_walk(long_context, idx, n_steps),
        # cfun=lambda idx, n_steps: ctxt_fn(var=v, mean=m, beta=b, dim=cdim, n_steps=n_steps)[0],
        stim_ntokens=stim_ntokens,
        sdim=sdim,
        n_context_steps=num_context_steps,
        training=True,
        neps_per_task=neps,
        sweight=sweight,
        cweight=cweight,
        nret=nret,
        hrate=hrate,
        stim_v=stim_v,
        sim_thresh=sim_thresh,
        verb=False
    )

    return evsc, ttype, fn_name

def calculate_dprime(evsc, ttype, model_name, fn_name, sweight, cweight, nret, hrate, fname='../Figures/sampling_model/'):
    taskL = [0, 1]
    rate_sum = np.zeros([len(taskL), 2, 4])

    dprimes = [None] * 4
    bias = [None] * 4
    hits = [None] * 4
    correct_rejections = [None] * 4
    labels = ["", "2b control", "2b lure", "4b control", "4b lure"]
    labels = ["2-back ctrl", "2-back lure", "3-back ctrl", "3-back lure"]

    idx = 0

    for task in taskL:
        task_score = evsc[task]
        # print(task_score.shape)
        # print("distribs of labels")
        # print(np.sum(task_score[:, 0]))
        for i in range(len(task_score)):
            ep = task_score[i]
            ttype_idx = int(ttype[task, i])
            if ep[0] == 1 and ep[1] == 1:
                rate_sum[task, ttype_idx, 0] += 1
            elif ep[0] == 1 and ep[1] == 0:
                rate_sum[task, ttype_idx, 1] += 1
            elif ep[0] == 0 and ep[1] == 1:
                rate_sum[task, ttype_idx, 2] += 1
            elif ep[0] == 0 and ep[1] == 0:
                rate_sum[task, ttype_idx, 3] += 1
            else:
                raise ValueError("unexpected set of values: " + str(ep))
        for ttype_idx in [0, 1]:
            n_ytrials = rate_sum[task, ttype_idx, 0] + rate_sum[task, ttype_idx, 2]
            n_ntrials = rate_sum[task, ttype_idx, 1] + rate_sum[task, ttype_idx, 3]
            # print("sums to")
            # print(n_ytrials + n_ntrials)
            # print("================")
            rate_sum[task, ttype_idx, 0] = rate_sum[task, ttype_idx, 0] / n_ytrials
            rate_sum[task, ttype_idx, 1] = rate_sum[task, ttype_idx, 1] / n_ntrials
            rate_sum[task, ttype_idx, 2] = rate_sum[task, ttype_idx, 2] / n_ytrials
            rate_sum[task, ttype_idx, 3] = rate_sum[task, ttype_idx, 3] / n_ntrials
            # REMOVE MANUAL MODIFICATION
            # rate_sum[task, ttype_idx] = rate_sum[task, ttype_idx] - 0.1
            # dprime = norm.ppf(rate_sum[task][0]) - norm.ppf(rate_sum[task][1])
            dprime, sensitivity = paper_dprime(rate_sum[task, ttype_idx][0], rate_sum[task, ttype_idx][1])
            dprimes[idx] = dprime
            bias[idx] = sensitivity
            hits[idx] = rate_sum[task, ttype_idx, 0]
            correct_rejections[idx] = rate_sum[task, ttype_idx, 3]
            idx += 1

    # param_string = model_name[model_name.find("seed") + 6:]
    # file_name = model_name[model_name.find("seed"):]
    # print(param_string)

    fig, axs = plt.subplots(2, 2, figsize=(10, 15))
    # fig.subplots_adjust(hspace=.75)
    x = range(4)
    width = 0.9

    # fig.suptitle(fn_name + "_sweight=" + str(sweight) + "_nret=" + str(nret) + "_hrate=" + str(hrate))
    fig.suptitle("Sampling Model Performance, N-Back", fontsize=20)

    titles = ["correct rejections", "hits", "sensitivity", "bias"]
    stats = [correct_rejections, hits, dprimes, bias]
    colors = ["C0", to_rgba("C0", 0.7), "C1", to_rgba("C1", 0.7)]

    for i in range(4):
        ax = axs[int(i / 2), i%2]
        for j in range(4):
            ax.bar(j, stats[i][j], width, label=labels[j], color=colors[j])
        ax.tick_params(axis='x', which="both", bottom=False)
        ax.set_title(titles[i], y=-0.12)
        if i < 2:
            ax.set_ylim(0, 1.1)
        else:
            ax.set_ylim(-5, 10)
            ax.axhline(0, color="black")
        if i == 0:
            ax.set_ylabel("Correct Rejection / Hit Rate")
        if i == 2:
            ax.set_ylabel("Signal Detection Measures")
        if i == 3:
            ax.legend()


    plt.savefig(fname + "split_dprime" + model_name + '_sw=' + str(sweight).replace(".", "") + "_nret=" + str(nret)  + "_hrate=" + str(hrate).replace(".", ""))
    plt.close('all')


def calculate_unified_dprime(evsc, ttype, model_name, fn_name, sweight, cweight, nret, hrate,  fname='../Figures/sampling_model/'):
    taskL = [0, 1]
    rate_sum = np.zeros([len(taskL), 4])

    dprimes = [None] * 2
    bias = [None] * 2
    hits = [None] * 2
    correct_rejections = [None] * 2
    labels = ["", "2back", "", "4back"]
    idx = 0

    for task in taskL:
        task_score = evsc[task]
        # print(task_score.shape)
        # print("distribs of labels")
        # print(np.sum(task_score[:, 0]))
        for i in range(len(task_score)):
            ep = task_score[i]
            ttype_idx = int(ttype[task, i])
            if ep[0] == 1 and ep[1] == 1:
                rate_sum[task, 0] += 1
            elif ep[0] == 1 and ep[1] == 0:
                rate_sum[task, 1] += 1
            elif ep[0] == 0 and ep[1] == 1:
                rate_sum[task, 2] += 1
            elif ep[0] == 0 and ep[1] == 0:
                rate_sum[task, 3] += 1
            else:
                raise ValueError("unexpected set of values: " + str(ep))
        n_ytrials = rate_sum[task, 0] + rate_sum[task, 2]
        n_ntrials = rate_sum[task, 1] + rate_sum[task, 3]
        # print("sums to")
    # print(n_ytrials + n_ntrials)
        # print("================")
        rate_sum[task, 0] = rate_sum[task, 0] / n_ytrials
        rate_sum[task, 1] = rate_sum[task, 1] / n_ntrials
        rate_sum[task, 2] = rate_sum[task, 2] / n_ytrials
        rate_sum[task, 3] = rate_sum[task, 3] / n_ntrials
        # dprime = norm.ppf(rate_sum[task][0]) - norm.ppf(rate_sum[task][1])
        dprime, sensitivity = paper_dprime(rate_sum[task][0], rate_sum[task][1])
        dprimes[idx] = dprime
        bias[idx] = sensitivity
        hits[idx] = rate_sum[task, 0]
        correct_rejections[idx] = rate_sum[task, 3]
        idx += 1

    # param_string = model_name[model_name.find("seed") + 6:]
    # file_name = model_name[model_name.find("seed"):]
    # print(param_string)

    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=.75)
    x = range(2)
    width = 1.0

    # fig.suptitle(fn_name + "_sweight=" + str(sweight) + "_nret=" + str(nret) + "_hrate=" + str(hrate))
    fig.suptitle("Full-Retrieval Model Performance, N-Back")


    titles = ["correct rejections", "hits", "sensitivity", "bias"]
    stats = [correct_rejections, hits, dprimes, bias]

    for i in range(4):
        ax = axs[int(i / 2), i%2]
        ax.bar(x, stats[i], width, color=["C0", "C1"])
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', rotation=45)
        ax.set_title(titles[i], y=-0.1)
        if i < 2:
            ax.set_ylim(0, 1.1)
        else:
            ax.set_ylim(-5, 10)
            ax.axhline(0, color="black")

    plt.savefig(fname + "unified_dprime" + model_name + '_sw=' + str(sweight).replace(".", "") + "_nret=" + str(nret)  + "_hrate=" + str(hrate).replace(".", ""))
    plt.close('all')

def sampling_dprime(net, model_name, ctxt_fn, m, v, b, d, fname='../Figures/sampling_model/'):
    sweight = 0.4
    cweight = 1.0 - sweight
    nret = 3
    hrate = 0.75
    stim_v = 0.25
    sim_thresh = 0.4
    evsc, ttype, fn_name = get_perf(net, model_name, ctxt_fn, m, v, b, d, sweight, cweight, nret, hrate, stim_v, sim_thresh)
    calculate_dprime(evsc, ttype, model_name, fn_name, sweight, cweight, nret, hrate, fname)
    calculate_unified_dprime(evsc, ttype, model_name, fn_name, sweight, cweight, nret, hrate, fname)

if __name__ == "__main__":
    # net_path = "/Users/maia/Projects/thesis-code/task_models/final_nets/trsc-ffwm-ttypenback_probs222--neps250000-seed1_fast_spherical_drift_d=25_m=025_v=005.pt"
    net_path = "/Users/maia/Projects/thesis-code/task_models/distributed_stimuli_nets/trsc-ffwm-ttypenback_probs422--neps400000-seed1_new_spherical_drift_d=25_m=025_v=005_stimset-v=025.pt"
    # net_path = "/Users/maia/Projects/thesis-code/task_models/interval_nets/trsc-ffwm-ttypenback_probs422--neps400000-seed1_new_spherical_drift_d=25_m=025_v=005_stimset-v=025200000.pt"
    # net_path = "/Users/maia/Projects/thesis-code/task_models/distributed_stimuli_nets/trsc-ffwm-ttypenback_probs422--neps400000-seed1_new_spherical_drift_d=25_m=04_v=0075_stimset-v=025.pt"
    net_path = "/Users/maia/Projects/thesis-code/task_models/2and3back/trsc-ffwm-ttypenback_probs422--neps400000-seed1_new_spherical_drift_d=25_m=025_v=005_stimset-v=075.pt"
    net = FFWM(indim, hiddim, add_noise=False)
    net.load_state_dict(tr.load(str(net_path)))
    d = 25
    m = 0.25
    v = 0.05
    b = 0.0
    ctxt_fn = fast_n_sphere
    sweight = 0.4
    cweight = 1.0 - sweight
    nret = 4
    hrate = 0.75
    stim_v = 0.25
    sim_thresh = 0.0

    model_name = net_path.split("/")[-1][:-3]
    model_name += "_pr-prelure=01_stimv0=25_sim-thresh=" + str(sim_thresh).replace(".", "") # + "_flipped-bits"
    print(model_name)

    evsc, ttype, fn_name = get_perf(net, model_name, ctxt_fn, m, v, b, d, sweight, cweight, nret, hrate, stim_v, sim_thresh)
    calculate_dprime(evsc, ttype, model_name, fn_name, sweight, cweight, nret, hrate, fname='../Figures/thresh-vs-hazard/')
    calculate_unified_dprime(evsc, ttype, model_name, fn_name, sweight, cweight, nret, hrate, fname='../Figures/thresh-vs-hazard/')

