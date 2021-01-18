import sys
import torch as tr
import numpy as np
from matplotlib import pyplot as plt
import glob
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colors import to_rgba

from drift_fns import return_sphere_context_supplier_from_params
from ModelsFFWM import *
from generate_stimset import gen_stims


## network parameters
c_dim = 25 # context dimensionality
s_dim = 20 # stimulus dimensionality
stim_ntokens = s_dim # number of unique stimuli
num_context_steps = 20 #
indim = 2*(c_dim + s_dim) # dimensionality of the input layer of the network
hiddim = c_dim * 4 # dimensionality of the hidden layer of the network
add_noise = False # should add Gaussian processing noise
flip_bits = True # should add random bit-flipping noise
task = "nback" # training task, either "nback" or "stern"

if task == 'nback':
  ssizeL = [2, 3] # set-size values for n-back
  sample_fn = lambda match,slure,clure: lambda S,C,N: sample_nback_trial_retrieval_sequence(S,
                                                                                            C,
                                                                                            N,
                                                                                            pr_match=match,
                                                                                            pr_stim_lure=slure,
                                                                                            pr_context_lure=clure)
elif task == 'stern':
  ssizeL = [4,9] # set-size values for sternberg
  sample_fn = lambda match,slure,clure: lambda S,C,N: sample_sternberg_trial_strictslure(S,C,N,
                pr_match=match,pr_stim_lure=slure,pr_context_lure=clure)


# Calculate the cosine similarity between two vectors
def cos_sim(a, b):
    return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))

# Flip bits on an input vector with probability pr_flip to mimic processing noise / distortion
def bit_flip_noise(vector, pr_flip):
    for ind in range(len(vector)):
        if np.random.uniform(0, 1) < pr_flip:
            if (vector[ind] > 0.5):
                vector[ind] = vector[ind] - 1
            else:
                vector[ind] = vector[ind] + 1
    return vector

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

# calculate dprime measures as in Kane et al
def paper_dprime(hrate, farate):
    hrate = clamp(hrate, 0.01, 0.99)
    farate = clamp(farate, 0.01, 0.99)
    dl = np.log(hrate * (1 - farate) / ((1 - hrate) * farate))
    c = 0.5 * np.log((1 - hrate) * (1 - farate) / (hrate * farate))
    return dl, c

def get_em_model_performance(net, taskL, cfun, neps_per_task, n_context_steps, stim_priority_weight, hrate, stim_v, sim_thresh, verb=True):
    """
    Train the model on a particular task, taking measures both of its cross-category training performance and of
    its d' characteristics
    """
    stimset = gen_stims(stim_v) # generate stimuli for use in training
    net.eval()
    score = -np.ones([len(taskL), neps_per_task, 2]) # array of performance information
    ttype = -np.ones([len(taskL), neps_per_task]) # array of trial-type information for d' measures
    for ep in range(neps_per_task):
        if verb and ep % (neps_per_task / 5) == 0:
          print(ep / neps_per_task)
        # resample context for this epoch
        cdrift = cfun(n_steps=n_context_steps, idx=ep)
        cdrift = tr.Tensor(cdrift)
        # interleave training for multiple task conditions each epoch
        for task_idx,(control_int,sample_trial_fn,setsize) in enumerate(taskL):
          # generate sample trace set
          stimseq,ytarget, ttype_idx = sample_trial_fn(stimset,cdrift,setsize)
          target = stimseq[0]
          sim_scores = []
          for memory in stimseq[1:]:
              # calculate a trace's retrieval priority based on weighted cosine similarity between the trace and the current trace
              cs_stim = cos_sim(target[0], memory[0])
              cs_ctxt = cos_sim(target[1], memory[1])
              sim = stim_priority_weight * cs_stim + (1 - stim_priority_weight) * cs_ctxt # weighting parameter
              sim_scores.append(sim)
          pred = 0
          # repeat as long as there are traces in the retrieval set
          for i in range(len(sim_scores)):
              max_sim_idx = np.argmax(np.array(sim_scores)) # pick the remaining trace with the highest similarity
              if (sim_scores[max_sim_idx] < sim_thresh):
                  break # break if the most-similar trace is under the threshold
              memory = stimseq[max_sim_idx + 1]
              inputL = [target[0],memory[0],target[1],memory[1]]
              yhat = net(inputL,control_bias_int=control_int)
              # predict a label for the trace
              pred = maxsoftmax(yhat).item()
              if pred == 1: # stop search if the model identifies a trace that evidences a positive trial
                  break
              if np.random.uniform() > (hrate): # stop search with probability hrate
                  break
              else:
                  sim_scores[max_sim_idx] = 0 # else zero out the retrieval priority of this trace
          score[task_idx, ep, 0] = pred
          score[task_idx, ep, 1] = ytarget.item()
          ttype[task_idx, ep] = ttype_idx
    return score, ttype

# Calculates Kane d' measures for a set of results and graphs them
def calculate_and_plot_dprime(evsc, ttype, model_name, stim_priority_weight, hrate, fname='../Figures/sampling_model/'):
    taskL = [0, 1]
    rate_sum = np.zeros([len(taskL), 2, 4])

    dprimes = [None] * 4
    bias = [None] * 4
    hits = [None] * 4
    correct_rejections = [None] * 4
    labels = ["2-back ctrl", "2-back lure", "3-back ctrl", "3-back lure"]

    idx = 0

    for task in taskL:
        task_score = evsc[task]
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
            rate_sum[task, ttype_idx, 0] = rate_sum[task, ttype_idx, 0] / n_ytrials
            rate_sum[task, ttype_idx, 1] = rate_sum[task, ttype_idx, 1] / n_ntrials
            rate_sum[task, ttype_idx, 2] = rate_sum[task, ttype_idx, 2] / n_ytrials
            rate_sum[task, ttype_idx, 3] = rate_sum[task, ttype_idx, 3] / n_ntrials
            dprime, sensitivity = paper_dprime(rate_sum[task, ttype_idx][0], rate_sum[task, ttype_idx][1])
            dprimes[idx] = dprime
            bias[idx] = sensitivity
            hits[idx] = rate_sum[task, ttype_idx, 0]
            correct_rejections[idx] = rate_sum[task, ttype_idx, 3]
            idx += 1

    fig, axs = plt.subplots(2, 2, figsize=(10, 15))
    width = 0.9

    fig.suptitle("d-prime measures of retrieval model performance", fontsize=20)

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

    plt.savefig(fname
                + "split_dprime" + model_name
                + '_sw=' + str(stim_priority_weight).replace(".", "")
                + "_hrate=" + str(hrate).replace(".", ""))

    plt.close('all')

# Calculates a set of d' measures for unified sequence types (control vs lure) and graphs them
def calculate_unified_dprime(evsc, ttype, model_name, stim_priority_weight, hrate,  fname='../Figures/sampling_model/'):
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
        rate_sum[task, 0] = rate_sum[task, 0] / n_ytrials
        rate_sum[task, 1] = rate_sum[task, 1] / n_ntrials
        rate_sum[task, 2] = rate_sum[task, 2] / n_ytrials
        rate_sum[task, 3] = rate_sum[task, 3] / n_ntrials
        dprime, sensitivity = paper_dprime(rate_sum[task][0], rate_sum[task][1])
        dprimes[idx] = dprime
        bias[idx] = sensitivity
        hits[idx] = rate_sum[task, 0]
        correct_rejections[idx] = rate_sum[task, 3]
        idx += 1

    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=.75)
    x = range(2)
    width = 1.0

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

    plt.savefig(fname
                + "unified_dprime" + model_name
                + '_sw=' + str(stim_priority_weight).replace(".", "")
                + "_hrate=" + str(hrate).replace(".", ""))

    plt.close('all')

def generate_dprimes(net, model_name, ctxt_fn, fname='../Figures/sampling_model/'):
    neps = 5000 # number of trials

    hrate = 0.75 # hazard rate
    sim_thresh = 0.4 # min similarity threshold
    rel_stim_weight_in_retrieval = 0.4 # relative weight of the stimulus (vs context) component in the similarity calculation for retrieval prioritization

    taskL = [0, 1]
    taskL_ev = [
        [taskL[0], sample_fn(0.25, 0.25, 0.25), ssizeL[0]],
        [taskL[1], sample_fn(0.25, 0.25, 0.25), ssizeL[1]]]

    evsc, ttype = get_em_model_performance(
        net,
        taskL_ev,
        cfun=ctxt_fn,
        n_context_steps=num_context_steps,
        neps_per_task=neps,
        stim_priority_weight=rel_stim_weight_in_retrieval,
        hrate=hrate,
        stim_v=stim_v,
        sim_thresh=sim_thresh,
        verb=False)

    calculate_and_plot_dprime(evsc, ttype, model_name, rel_stim_weight_in_retrieval, hrate, fname)
    calculate_unified_dprime(evsc, ttype, model_name, rel_stim_weight_in_retrieval, hrate, fname)

if __name__ == "__main__":
    # net_path = "/Users/maia/Projects/thesis-code/task_models/final_nets/trsc-ffwm-ttypenback_probs222--neps250000-seed1_fast_spherical_drift_d=25_m=025_v=005.pt"
    # net_path = "/Users/maia/Projects/thesis-code/task_models/distributed_stimuli_nets/trsc-ffwm-ttypenback_probs422--neps400000-seed1_new_spherical_drift_d=25_m=025_v=005_stimset-v=025.pt"
    # net_path = "/Users/maia/Projects/thesis-code/task_models/interval_nets/trsc-ffwm-ttypenback_probs422--neps400000-seed1_new_spherical_drift_d=25_m=025_v=005_stimset-v=025200000.pt"
    # net_path = "/Users/maia/Projects/thesis-code/task_models/distributed_stimuli_nets/trsc-ffwm-ttypenback_probs422--neps400000-seed1_new_spherical_drift_d=25_m=04_v=0075_stimset-v=025.pt"
    net_path = "/Users/maia/Projects/thesis-code/task_models/2and3back/trsc-ffwm-ttypenback_probs422--neps400000-seed1_new_spherical_drift_d=25_m=025_v=005_stimset-v=075.pt"

    net = FFWM(indim, hiddim, add_noise=False)
    net.load_state_dict(tr.load(str(net_path)))

    d = 25
    m = 0.25
    v = 0.05

    ctxt_fn = return_sphere_context_supplier_from_params(d, m, v)

    model_name = net_path.split("/")[-1][:-3]
    print(model_name)

    run_dprimes(net, model_name, ctxt_fn, "..")

