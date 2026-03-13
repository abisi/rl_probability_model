import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def LRate(x, q):
    return q * x if x >= 0 else x

def run_learning(x1, x2, Q_init, Qz_init, a, b, c, v, Nmax1):
    Q = Q_init.copy()
    Qz = Qz_init.copy()
    sqrt2 = np.sqrt(2)

    p1 = []
    p2 = []

    for _ in range(Nmax1):
        dQx1 = np.dot(Q - Qz, x1)
        dQx2 = np.dot(Q - Qz, x2)

        p1.append((1 + erf(dQx1 / sqrt2)) / 2)
        p2.append((1 + erf(dQx2 / sqrt2)) / 2)

        Q += Q * (b / 2 * (LRate(-1 - a * dQx2, v) * p2[-1] * x2 +
                           LRate(1 - a * dQx1, v) * p1[-1] * x1)) + c * (Qz - Q)
        Qz += Qz * (-b / 2 * (LRate(-1 - a * dQx2, v) * p2[-1] * x2 +
                             LRate(1 - a * dQx1, v) * p1[-1] * x1)) + c * (Q - Qz)

    p1 = np.array(p1)
    p2 = np.array(p2)
    p_combined = (p1 + 1 - p2) / 2

    return p_combined

def run_rl_model(Nmax1=4000, dt=5, reward_new_s_plus=True):
    # Parameters
    Qo_strong = np.array([0.0001, 0.0001, 0.7])
    Qzo_strong = np.array([0.0001, 0.0001, 0.01])

    Qo_weak = np.array([0.00005, 0.00005, 0.7])
    Qzo_weak = np.array([0.00005, 0.00005, 0.01])

    b = 0.01 * dt
    a = 0.6195
    c = 0
    v = 6
    as_param = 2
    Nmax1 = Nmax1 // dt

    results = {}

    # Trial Type 1: First S+ more salient
    x1 = np.array([0, 1, 1])                # S+
    x2 = np.array([as_param, 0, 1])         # S−
    results['type1'] = run_learning(x1, x2, Qo_strong, Qzo_strong, a, b, c, v, Nmax1)

    # Trial Type 2: First S+ less salient
    x1b = np.array([0, as_param, 1])        # S+
    x2b = np.array([1, 0, 1])               # S−
    results['type2'] = run_learning(x1b, x2b, Qo_strong, Qzo_strong, a, b, c, v, Nmax1)

    # Trial Type 3: New S+ (rewarded or not)
    x1c = np.array([1, as_param, 1])        # New S+
    if reward_new_s_plus:
        x2c = np.array([as_param, 1, 1])    # Random S− input
    else:
        x2c = np.array([1, as_param, 1])    # Identical to new S+, not rewarded
    results['type3'] = run_learning(x1c, x2c, Qo_weak, Qzo_weak, a, b, c, v, Nmax1)

    return results, Nmax1

def plot_learning_curves(results_A, results_B, Nmax1, dt=5):
    z = 20 // dt
    nx = np.arange(int(1.5 * z), Nmax1)
    time = nx * dt

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axs[0].plot(time, results_A['type1'][nx], 'r', label='First S+ (Type 1)')
    axs[0].plot(time, results_A['type2'][nx], 'b', label='First S+ (Type 2)')
    axs[0].plot(time, results_A['type3'][nx], 'g', label='New S+ (Type 3)')
    axs[0].set_title("Mouse A (New S+ is Rewarded)")
    axs[0].set_xlabel("Trial")
    axs[0].set_ylabel("Probability Correct")
    axs[0].set_ylim([0, 1])
    axs[0].legend()

    axs[1].plot(time, results_B['type1'][nx], 'r', label='First S+ (Type 1)')
    axs[1].plot(time, results_B['type2'][nx], 'b', label='First S+ (Type 2)')
    axs[1].plot(time, results_B['type3'][nx], 'g', label='New S+ (Type 3)')
    axs[1].set_title("Mouse B (New S+ is Not Rewarded)")
    axs[1].set_xlabel("Trial")
    axs[1].legend()

    plt.suptitle("Learning Curves: Rewarded vs. Unrewarded New S+")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    dt = 5
    results_A, Nmax1 = run_rl_model(dt=dt, reward_new_s_plus=True)
    results_B, _ = run_rl_model(dt=dt, reward_new_s_plus=False)
    plot_learning_curves(results_A, results_B, Nmax1, dt=dt)
