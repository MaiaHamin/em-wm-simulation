import torch as tr
import numpy as np

from training_code.generate_figures import plot_train_accuracy, plot_accuracy_by_trial_type
from training_code.nback_stern_tasks import single_nback_comparison, single_sternberg_comparison

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

def run_model_for_epochs(net, taskL, ctxt_fn, stim_fn, training, neps_per_task, n_ctxt_steps, verb=True):
  if training:
    net.train()
  else:
    net.eval()
  lossop = tr.nn.CrossEntropyLoss()
  optiop = tr.optim.Adam(net.parameters(), lr=0.001)
  score = -np.ones([len(taskL), neps_per_task])
  ttype = -np.ones([len(taskL), neps_per_task])
  for ep in range(neps_per_task):
    if verb and ep % (neps_per_task / 5) == 0:
      print(ep / neps_per_task)
    # resample stim and context on each ep
    stimset = stim_fn()
    cdrift = ctxt_fn(n_steps=n_ctxt_steps)
    cdrift = tr.Tensor(cdrift)
    # interleave train on every task per epoch
    for task_idx,(control_int,sample_trial_fn,setsize) in enumerate(taskL): 
      # use the input function to generate a trial sample
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

maxsoftmax = lambda x: tr.argmax(tr.softmax(x,-1),-1).squeeze()

def train_net_and_plot_accuracy(task, ctxt_fn, stim_fn, c_dim, s_dim, neps, n_ctxt_steps, pr_match, pr_slure, pr_clure, seed, model_path, figure_path):
  np.random.seed(seed)
  tr.random.manual_seed(seed)

  ## net params
  indim = 2 * (c_dim + s_dim)
  hiddim = s_dim * 4

  if task == 'nback':
    ssizeL = [2, 3] # [1, 3]
    sample_fn = lambda match,slure,clure: lambda S,C,N: single_nback_comparison(S,C,N,
                  pr_match=match,pr_stim_lure=slure,pr_context_lure=clure
                  )
  elif task == 'stern':
    ssizeL = [4,9]
    sample_fn = lambda match,slure,clure: lambda S,C,N: single_sternberg_comparison(S,C,N,
                  pr_match=match,pr_stim_lure=slure,pr_context_lure=clure
                  )

  taskintL = [0,1]
  taskL_tr = [
    [taskintL[0],sample_fn(pr_match,pr_slure,pr_clure),ssizeL[0]],
    [taskintL[1],sample_fn(pr_match,pr_slure,pr_clure),ssizeL[1]],
  ]

  # init net
  net = FFWM(indim,hiddim)

  # train net
  score_tr, ttype = run_model_for_epochs(
    net,
    taskL_tr,
    ctxt_fn=ctxt_fn,
    stim_fn = stim_fn,
    training=True,
    neps_per_task=neps,
    n_ctxt_steps=n_ctxt_steps)

  np.save(model_path + "/train-score", score_tr)
  tr.save(net.state_dict(), model_path + "/trained-net.pt")

  plot_train_accuracy(score_tr, ttype, ssizeL, figure_path)

  evac = eval_by_ttype(net, ctxt_fn, sample_fn, ssizeL, taskintL, neps=1000)
  plot_accuracy_by_trial_type(evac, taskintL, ssizeL, figure_path)

  return net

def eval_by_ttype(net, ctxt_fn, sample_fn, ssize, task_int, neps):
  """ eval on given task for separate trial types
  returns evac on (match,nomatch,slure,clure)
  """

  taskL_ev = [
    [task_int, sample_fn(1, 0, 0), ssize],
    [task_int, sample_fn(0, 0, 0), ssize],
    [task_int, sample_fn(0, 1, 0), ssize],
    [task_int, sample_fn(0, 0, 1), ssize]
  ]

  evsc, ttype = run_model_for_epochs(
    net, taskL_ev,
    ctxt_fn=ctxt_fn,
    training=False,
    neps_per_task=neps,
    verb=False
  )

  evac = evsc.mean(1)
  return evac


