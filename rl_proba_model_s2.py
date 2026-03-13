import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def LRate(x, v):
    return v * x if x > 0 else x

def run_learning(x1, x2, Q_init, Qz_init, a, b, c, v, Nmax1):
    Q = Q_init.copy()
    Qz = Qz_init.copy()
    sqrt2 = np.sqrt(2)

    p1 = []
    p2 = []
    Q_history = []
    Qz_history = []

    for i in range(Nmax1):
        dQx1 = np.dot(Q - Qz, x1)
        dQx2 = np.dot(Q - Qz, x2)

        p1.append(0.5 * (1 + erf(dQx1 / sqrt2)))
        p2.append(0.5 * (1 + erf(dQx2 / sqrt2)))

        deltaQ = (
            + LRate(-1 - a * dQx2, v) * p2[-1] * x2
            + LRate(+1 - a * dQx1, v) * p1[-1] * x1
        )

        Q = Q * (1 + b / 2 * deltaQ) + c * Qz - c * Q
        Qz = Qz * (1 - b / 2 * deltaQ) + c * Q - c * Qz

        Q_history.append(Q.copy())
        Qz_history.append(Qz.copy())

    return {
        'p1': np.array(p1),  # S+
        'p2': np.array(p2),  # S-
        'correct': (np.array(p1) + (1 - np.array(p2))) / 2
    }, np.array(Q_history), np.array(Qz_history)

def run_rl_model(Nmax=4000, dt=5, reward_whisker=True):
    stim_auditory = np.array([1, 1, 0, 0])
    stim_whisker = np.array([1, 0, 1, 0])
    stim_nostim = np.array([1, 0, 0, 1])

    if reward_whisker:
        Q_init = np.array([0.5, 1.5, 0.3, 0.01])
        Qz_init = np.array([0.05, 0.1, 0.1, 0.01])
        v = 6
    else:
        bias_transfer = 0.5
        Q_init = np.array([0.5, 1.5, 1.5 * bias_transfer, 0.01])
        Qz_init = np.array([0.05, 0.1, 0.1, 0.01])
        v = 10  # larger error signal for unexpected no-reward

    b = 0.01 * dt
    a = 0.6195
    c = 0
    Nmax1 = Nmax // dt

    results = {}

    results['auditory'], Q_aud, _ = run_learning(
        stim_auditory, stim_nostim, Q_init, Qz_init, a, b, c, v, Nmax1)

    if reward_whisker:
        results['whisker'], Q_wh, _ = run_learning(
            stim_whisker, stim_nostim, Q_init, Qz_init, a, b, c, v, Nmax1)
    else:
        results['whisker'], Q_wh, _ = run_learning(
            stim_nostim, stim_whisker, Q_init, Qz_init, a, b, c, v, Nmax1)

    results['weights_auditory'] = Q_aud
    results['weights_whisker'] = Q_wh

    return results, Nmax1

def plot_lick_probabilities(results_rewarded, results_nonrewarded, dt, Nmax1):
    t = np.arange(Nmax1) * dt

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, results, title in zip(
        axs,
        [results_rewarded, results_nonrewarded],
        ["Rewarded Whisker Mouse", "Non-Rewarded Whisker Mouse"]
    ):
        ax.plot(t, results['auditory']['p1'], 'b', label='Auditory S+')
        ax.plot(t, results['whisker']['p1'], 'g', label='Whisker S+')
        ax.plot(t, 1 - results['auditory']['p2'], 'r--', label='No Stim (S−)')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("P(Lick)")
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(True)

    axs[1].set_xlabel("Trial")
    plt.tight_layout()
    plt.show()

def plot_weight_dynamics(results, dt, Nmax1):
    t = np.arange(Nmax1) * dt
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for ax, weights, title in zip(
        axs,
        [results['weights_auditory'], results['weights_whisker']],
        ['Weights: Auditory S+', 'Weights: Whisker S+']
    ):
        ax.plot(t, weights[:, 1], label='Auditory', color='b')
        ax.plot(t, weights[:, 2], label='Whisker', color='g')
        ax.plot(t, weights[:, 3], label='NoStim', color='r')
        ax.plot(t, weights[:, 0], label='General (C)', color='k', linestyle='--')
        ax.set_ylabel("Weight")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    axs[1].set_xlabel("Trial")
    plt.tight_layout()
    plt.show()

# ---- Run the simulations ----

dt = 5
Nmax = 4000

results_rewarded, Nmax1 = run_rl_model(Nmax=Nmax, dt=dt, reward_whisker=True)
results_nonrewarded, _ = run_rl_model(Nmax=Nmax, dt=dt, reward_whisker=False)

plot_lick_probabilities(results_rewarded, results_nonrewarded, dt, Nmax1)
plot_weight_dynamics(results_rewarded, dt, Nmax1)
plot_weight_dynamics(results_nonrewarded, dt, Nmax1)
