from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

# plot training accuracy
def plot_train_accuracy(score_tr,
                        ttype,
                        ssizeL,
                        figure_path):

  task_labels = ["setsize " + str(ssize) for ssize in ssizeL]
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
      lt = task_labels[i] + " " + labels[tt]
      plt.plot(ac, color=task_color[tt], label=lt)
  plt.legend(loc="best")
  plt.ylim(0, 1)
  plt.ylabel('Train accuracy')
  plt.savefig(figure_path + "/train-accuracy")
  plt.close('all')

# eval on neps_ev iterations of each trial type and plot the accuracy for each
def plot_accuracy_by_trial_type(evac,
                                taskintL,
                                ssizeL,
                                figure_path):
  for tidx, (task_int, ssize) in enumerate(zip(taskintL, ssizeL)):
    plt.title('Accuracy by trial type')
    plt.bar(np.arange(4) + (.45 * tidx), evac[tidx], width=.45, label="setsize:" + str(ssize))
  plt.legend()
  plt.xticks(range(4), ['match', 'nomatch', 'slure', 'clure'])
  plt.savefig(figure_path + "/trial-type-accuracy")
  plt.close('all')


# calculate dprime measures as in Kane et al
def paper_dprime(hrate, farate):
    hrate = clamp(hrate, 0.01, 0.99)
    farate = clamp(farate, 0.01, 0.99)
    dl = np.log(hrate * (1 - farate) / ((1 - hrate) * farate))
    c = 0.5 * np.log((1 - hrate) * (1 - farate) / (hrate * farate))
    return dl, c

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

# Calculates Kane d' measures for a set of results and graphs them
def calculate_and_plot_dprime(evsc, ttype, dprime_figure_path):
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

    plt.savefig(dprime_figure_path)

    plt.close('all')

# Calculates a set of d' measures for unified sequence types (control target & foil, lure target & foil) and plots them
def calculate_unified_dprime(evsc, ttype, stim_priority_weight, hrate,  figure_path):
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

    plt.savefig(figure_path + "unified-sequence-dprime"
                + '_sw=' + str(stim_priority_weight).replace(".", "")
                + "_hrate=" + str(hrate).replace(".", ""))

    plt.close('all')