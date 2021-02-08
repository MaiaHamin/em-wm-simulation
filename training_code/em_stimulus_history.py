import torch as tr
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from training_code.em_trace_model import return_sphere_context_supplier_from_params, gen_stims
from training_code.nback_stern_tasks import sample_nback_retrieval_set_for_trial_sequence
from training_code.train_wm_model import maxsoftmax
from training_code.generate_figures import calculate_and_plot_dprime

# Calculate the cosine similarity between two vectors
def cos_sim(a, b):
    return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))

def get_em_model_performance(net, taskL, ctxt_fn, stim_fn, neps_per_task, n_context_steps, stim_priority_weight, hrate, sim_thresh):
    """
    Train the model on a particular task, taking measures both of its cross-category training performance and of
    its d' characteristics
    """
    net.eval()
    score = -np.ones([len(taskL), neps_per_task, 2]) # array of performance information
    ttype = -np.ones([len(taskL), neps_per_task]) # array of trial-type information for d' measures
    for ep in range(neps_per_task):
        # resample context for this epoch
        stimset = tr.Tensor(stim_fn())
        cdrift = ctxt_fn(n_steps=n_context_steps)
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

def simulate_em_and_plot_dprimes(net, ctxt_fn, stim_fn, neps_per_task, pr_match, pr_slure, pr_clure, n_context_steps, stim_priority_weight, hrate, sim_thresh, figure_path):
    ssizeL = [2, 3]
    sample_fn = lambda match, slure, clure: lambda S, C, N: sample_nback_retrieval_set_for_trial_sequence(S, C, N,
                                                                                    pr_match=match,
                                                                                    pr_stim_lure=slure,
                                                                                    pr_context_lure=clure
                                                                                    )
    taskintL = [0, 1]
    taskL = [
        [taskintL[0], sample_fn(pr_match, pr_slure, pr_clure), ssizeL[0]],
        [taskintL[1], sample_fn(pr_match, pr_slure, pr_clure), ssizeL[1]],
    ]

    score, ttype = get_em_model_performance(net, taskL, ctxt_fn, stim_fn, neps_per_task, n_context_steps,
                                            stim_priority_weight, hrate, sim_thresh)

    dprime_fig_path = figure_path + ("/dprime" + '_sw=' + str(stim_priority_weight) + "_hrate=" + str(hrate)).replace(".", "")

    calculate_and_plot_dprime(score, ttype, dprime_figure_path=dprime_fig_path)

