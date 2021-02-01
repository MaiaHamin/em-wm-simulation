import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from training_code.em_trace_model import spherical_drift, gen_stims
from training_code.train_wm_model import train_net_and_plot_accuracy
from training_code.em_stimulus_history import simulate_em_and_plot_dprimes

# change this variable to change the task between nback and stern
task = "nback"  # "stern"

# function and parameters for the drifting context
ctxt_fn = spherical_drift
drift_parameters = [(0.25, 0.0), (0.25, 0.05), (0.25, 0.075), (0.4, 0.075)]
ctxt_d = 25
n_ctxt_steps=20

# function which returns a stimulus - here, always an identity matrix, but could be redrawn from a distribution each time, e.g.
stim_d = 20
stim_fn = lambda: gen_stims(stim_d)

# probabilities of different training conditions: both match, stimulus lure, context lure, neither match
pr_match, pr_slure, pr_clure, pr_nomatch = 0.4, 0.2, 0.2, 0.2

# priority weighting of stimulus similarity (vs context similarity) in EM retrieval
stim_retrieval_weight = 0.4
# hazard rate - likelihood of terminating memory search at each step
h_rate = 0.6
# similarity threshold cutoff - min cosine similarity before search is terminated deterministically
similarity_threshold = 0.5

n_training_eps=10000

for seed in [1, 2]:
  for (mean, var) in drift_parameters:

      # Create descriptive model name & associated directories
      model_name = 'ffwm_task-%s_training-probs-%i-%i-%i-%i_neps-%i_drift-params-%i-%i_seed%i' % (
          task, pr_match, pr_slure, pr_clure, pr_nomatch, mean, var, n_training_eps, seed)

      model_name = model_name.replace(".", "")

      # create directories for the model and figures
      figure_path = "../figures/" + model_name
      if not os.path.exists(figure_path):
          os.makedirs(figure_path)
      model_path = "../trained-models/" + model_name
      if not os.path.exists(model_path):
          os.makedirs(model_path)

      print("Model path: " + model_path)
      print("Figure path: " + figure_path)

      print("Training WM... ")

      stimset = gen_stims(stim_d)

      net = train_net_and_plot_accuracy(task, ctxt_fn, stim_fn, c_dim=ctxt_d, s_dim=stim_d, neps=n_training_eps, n_ctxt_steps=n_ctxt_steps,
                                        pr_match=pr_match, pr_slure=pr_slure, pr_clure=pr_clure, seed=seed,
                                        model_path=model_path, figure_path=figure_path)

      if task == "nback":
          print("Simulating EM...")
          simulate_em_and_plot_dprimes(net, ctxt_fn, stim_fn,
                                       neps_per_task=10000, pr_match=pr_match,
                                        pr_slure=pr_slure, pr_clure=pr_clure, n_context_steps=n_ctxt_steps,
                                       stim_priority_weight=stim_retrieval_weight,
                                       hrate = h_rate,
                                       sim_thresh = similarity_threshold,
                                       figure_path = figure_path)
