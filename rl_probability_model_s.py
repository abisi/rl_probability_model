import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def LRate(x, q):
    return q * x if x >= 0 else x

def rl_probability_model_s(Nmax1=4000, dt=5, x=None, plot=True):
    if x is None:
        Qo  = np.array([0.0001, 0.0001, 0.7])
        Qzo = np.array([0.0001, 0.0001, 0.01])
        b = 0.01 * dt
        a = 0.6195
        c = 0
        v = 6
        as_param = 2
    else:
        Qo  = np.array([x[0], x[0], x[1]])
        Qzo = np.array([x[0], x[0], x[2]])
        a = x[3]
        b = x[4] * dt
        as_param = x[5]
        v = x[6]
        c = 0 * b
        Nmax1 = int(Nmax1 / dt)

    sqrt2 = np.sqrt(2)

    # First condition: S− more salient
    x1 = np.array([0, 1, 1])
    x2 = np.array([as_param, 0, 1])
    Q = Qo.copy()
    Qz = Qzo.copy()

    p1 = []
    p2 = []
    g1 = []
    g2 = []

    for _ in range(Nmax1):
        dQx1 = np.dot(Q - Qz, x1)
        dQx2 = np.dot(Q - Qz, x2)

        p1.append((1 + erf(dQx1 / sqrt2)) / 2)
        p2.append((1 + erf(dQx2 / sqrt2)) / 2)

        Q += Q * (b/2 * (LRate(-1 - a * dQx2, v) * p2[-1] * x2 + LRate(1 - a * dQx1, v) * p1[-1] * x1)) + c * (Qz - Q)
        Qz += Qz * (-b/2 * (LRate(-1 - a * dQx2, v) * p2[-1] * x2 + LRate(1 - a * dQx1, v) * p1[-1] * x1)) + c * (Q - Qz)

        g1.append(Q.copy())
        g2.append(Qz.copy())

    dQxc1 = np.dot(Q - Qz, np.array([0, 0, 1]))
    pc1 = (1 + erf(dQxc1 / sqrt2)) / 2

    # Second condition: S− less salient
    x1 = np.array([0, as_param, 1])
    x2 = np.array([1, 0, 1])
    Q = Qo.copy()
    Qz = Qzo.copy()

    p1b = []
    p2b = []
    g1b = []
    g2b = []

    for _ in range(Nmax1):
        dQx1 = np.dot(Q - Qz, x1)
        dQx2 = np.dot(Q - Qz, x2)

        p1b.append((1 + erf(dQx1 / sqrt2)) / 2)
        p2b.append((1 + erf(dQx2 / sqrt2)) / 2)

        Q += Q * (b/2 * (LRate(-1 - a * dQx2, v) * p2b[-1] * x2 + LRate(1 - a * dQx1, v) * p1b[-1] * x1)) + c * (Qz - Q)
        Qz += Qz * (-b/2 * (LRate(-1 - a * dQx2, v) * p2b[-1] * x2 + LRate(1 - a * dQx1, v) * p1b[-1] * x1)) + c * (Q - Qz)

        g1b.append(Q.copy())
        g2b.append(Qz.copy())

    dQxc2 = np.dot(Q - Qz, np.array([0, 0, 1]))
    pc2 = (1 + erf(dQxc2 / sqrt2)) / 2

    p1 = np.array(p1)
    p2 = np.array(p2)
    p1b = np.array(p1b)
    p2b = np.array(p2b)

    p_combined_1 = (p1 + 1 - p2) / 2
    p_combined_2 = (p1b + 1 - p2b) / 2

    if plot:
        z = 20 // dt
        nx = np.arange(int(1.5*z), Nmax1)

        fig, axs = plt.subplots(4, 2, figsize=(12, 10))

        axs[0, 0].plot(nx * dt, p1[nx], 'r', label='S+')
        axs[0, 0].plot(nx * dt, 1 - p2[nx], 'c', label='1 - S−')
        axs[0, 0].plot(nx * dt, p_combined_1[nx], 'k', label='Avg Correct')
        axs[0, 0].set_ylim([0, 1])
        axs[0, 0].set_ylabel('Proba correct')
        axs[0, 0].legend()

        axs[0, 1].plot(nx * dt, p1b[nx], 'r', label='S+')
        axs[0, 1].plot(nx * dt, 1 - p2b[nx], 'c', label='1 - S−')
        axs[0, 1].plot(nx * dt, p_combined_2[nx], 'k', label='Avg Correct')
        axs[0, 1].set_ylim([0, 1])
        axs[0, 1].set_ylabel('Proba correct')
        axs[0, 1].legend()

        axs[1, 0].plot(np.arange(Nmax1)*dt, np.array(g1)[:, 0], label='w0')
        axs[1, 0].plot(np.arange(Nmax1)*dt, np.array(g1)[:, 1], label='w1')
        axs[1, 0].plot(np.arange(Nmax1)*dt, np.array(g1)[:, 2], label='w2')
        axs[1, 0].set_ylim([0, 3])
        axs[1, 0].set_ylabel('Synaptic weight')
        axs[1, 0].legend()

        axs[1, 1].plot(np.arange(Nmax1)*dt, np.array(g1b)[:, 0], label='w0')
        axs[1, 1].plot(np.arange(Nmax1)*dt, np.array(g1b)[:, 1], label='w1')
        axs[1, 1].plot(np.arange(Nmax1)*dt, np.array(g1b)[:, 2], label='w2')
        axs[1, 1].set_ylim([0, 3])
        axs[1, 1].set_ylabel('Synaptic weight')
        axs[1, 1].legend()

        axs[2, 0].plot(np.arange(Nmax1)*dt, np.array(g2)[:, 0], label='z0')
        axs[2, 0].plot(np.arange(Nmax1)*dt, np.array(g2)[:, 1], label='z1')
        axs[2, 0].plot(np.arange(Nmax1)*dt, np.array(g2)[:, 2], label='z2')
        axs[2, 0].set_ylim([0, 3])
        axs[2, 0].set_ylabel('Synaptic weight')
        axs[2, 0].legend()

        axs[2, 1].plot(np.arange(Nmax1)*dt, np.array(g2b)[:, 0], label='z0')
        axs[2, 1].plot(np.arange(Nmax1)*dt, np.array(g2b)[:, 1], label='z1')
        axs[2, 1].plot(np.arange(Nmax1)*dt, np.array(g2b)[:, 2], label='z2')
        axs[2, 1].set_ylim([0, 3])
        axs[2, 1].set_ylabel('Synaptic weight')
        axs[2, 1].legend()

        axs[3, 0].bar([0, 1, 2], [p1[-1], p2[-1], pc1], tick_label=['S+', 'S−', 'C'])
        axs[3, 0].set_ylabel('Proba licking')

        axs[3, 1].bar([0, 1, 2], [p1b[-1], p2b[-1], pc2], tick_label=['S+', 'S−', 'C'])
        axs[3, 1].set_ylabel('Proba licking')

        plt.tight_layout()
        plt.show()

    return (p_combined_1, p_combined_2)

# Run if script is executed
if __name__ == '__main__':
    rl_probability_model_s()
