import os
from cProfile import label

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import erf

import plotting_utils as putils

# Asymmetric learning rate function, faster for unexpected rewards
def LRate(x, v, whisker_reward=True):
    if whisker_reward:
        return v * x if x < 0 else x
        #return v * x if x < 0 else x
    else:
        #return v * x if x < 0 else x
        return v * x if x > 0 else x


# RL model with one S+ and one S−
def rl_model_proba(Nmax=4000, dt=5, whisker_reward=True, **kwargs):

    if kwargs is not None:
        print('Params:', kwargs)

    # Inputs: [auditory,whisker,no_stim,C]
    stim_auditory = np.array([1,0,0,1])  # S+ (auditory)
    stim_whisker = np.array([0,1,0,1])  # S+ (whisker)
    stim_no_stim = np.array([0,0,1,1])  # S− (no stimulation)

    # Initial weights : higher for auditory, lower for whisker
    Q_exc_init = np.array([1.1, 0.01, 0.001, 0.5]) # excitatory
    #Q_exc_init = np.array([kwargs['Q_ea'], kwargs['Q_ea']/100, 0.001, 0.5]) # excitatory #TODO: continue varying pretrianing strength at init.
    #Q_exc_init = np.array([1.1, kwargs['Q_ew'], 0.001, 0.5]) # excitatory #TODO: continue varying pretrianing strength at init.
    Q_inh_init = np.array([0.00001, 0.001, 0.1, 0.9]) # inhibitory
    #Q_inh_init = np.array([0.00001, 0.001, kwargs['Q_in'], 0.9]) # inhibitory

    # Initial weights : as if no pretraining, all weights are same
    #Q_exc_init = np.array([0.001, 0.001, 0.001, 0.5]) # excitatory
    #Q_inh_init = np.array([0.001, 0.001, 0.1, 0.9]) # inhibitory

    # Initial weights : higher for auditory, lower for whisker
    #Q_exc_init = np.array([0.9, 0.1, 0.001, 0.5]) # excitatory KEEP
    #Q_inh_init = np.array([0.00001, 0.001, 0.01, 0.9]) # inhibitory KEEP


    # Parameters
    Nmax1 = Nmax // dt
    sqrt2 = np.sqrt(2)
    b = 0.025 * dt # learning rate
    a = 0.6195 # controls asymptotic perf
    c = 0
    v = 2 # asymmetric learning rule
    if whisker_reward:
        reward = 1
    else:
        reward = -1

    Q_exc = Q_exc_init.copy()
    Q_inh = Q_inh_init.copy()

    # Init. collections for results
    p_aud = [] # auditory P(lick)
    p_whisk = []
    p_no_stim = []

    Q_exc_hist = [] # Q_exc history, synaptic weights
    Q_inh_hist = []

    for i in range(Nmax1):

        # Stop auditory trials
        if i > int(Nmax1 / 2):
        #    stim_auditory = np.array([0, 0, 0, 1])
            aud_reward = -1
        else:
            aud_reward = 1

        # Decision values
        dQ_aud = np.dot(Q_exc - Q_inh, stim_auditory)
        dQ_whisk = np.dot(Q_exc - Q_inh, stim_whisker)
        dQ_no_stim = np.dot(Q_exc - Q_inh, stim_no_stim)

        # Probabilities of licking, at time t
        p1 = 0.5 * (1 + erf(dQ_aud / sqrt2))  # Aud. P(lick), ergodic approximation - see PNAS paper
        p2 = 0.5 * (1 + erf(dQ_whisk / sqrt2))  # Whisker P(lick)
        p3 = 0.5 * (1 + erf(dQ_no_stim / sqrt2))  # No Stim. P(lick)

        p_aud.append(p1)
        p_whisk.append(p2)
        p_no_stim.append(p3)

        # Update rule: auditory trial
        Q_exc  = Q_exc * (1 + b / 2 * (
            LRate(aud_reward - a * dQ_aud, v=1, whisker_reward=False) * p1 * stim_auditory
        )) + c * Q_inh - c * Q_exc

        Q_inh = Q_inh * (1 - b / 2 * (
            LRate(aud_reward - a * dQ_aud, v=1, whisker_reward=False) * p1 * stim_auditory
        )) + c * Q_exc - c * Q_inh

        # Update rule: whisker trial
        #if whisker_reward:
        #    reward = 1 if np.random.rand() < kwargs['wh_reward_proba'] else 0 # probabilistic reward
        Q_exc = Q_exc * (1 + b / 2 * (
                LRate(reward - a * dQ_whisk, v=v, whisker_reward=whisker_reward) * p2 * stim_whisker
        )) + c * Q_inh - c * Q_exc

        Q_inh = Q_inh * (1 - b / 2 * (
                LRate(reward - a * dQ_whisk, v=v, whisker_reward=whisker_reward) * p2 * stim_whisker
        )) + c * Q_exc - c * Q_inh

        # Update rule: no stim trial
        Q_exc = Q_exc * (1 + b / 2 * (
                LRate(-1 - a * dQ_no_stim, v=v, whisker_reward=whisker_reward) * p3 * stim_no_stim
        )) + c * Q_inh - c * Q_exc

        Q_inh = Q_inh * (1 - b / 2 * (
                LRate(-1 - a * dQ_no_stim, v=v, whisker_reward=whisker_reward) * p3 * stim_no_stim
        )) + c * Q_exc - c * Q_inh

        # Save histories
        Q_exc_hist.append(Q_exc.copy())
        Q_inh_hist.append(Q_inh.copy())

    return np.array(p_aud), np.array(p_whisk), np.array(p_no_stim), np.array(Q_exc_hist), np.array(Q_inh_hist)

# ---------
# Run model
# ---------
dt = 2
Nmax = 1000
#p1, p2, prob_correct, Q_hist, Qz_hist = rl_model_single_S(Nmax=Nmax, dt=dt)

# Sweep parameter range
#param_name = 'wh_reward_proba'
#param_values = np.linspace(0, 1, 11)v

param_name = 'Q_ea'
param_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.1]
param_name = 'Q_in'
param_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.1]
param_name = 'Q_ew'
param_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.1]

print(f'Fitting {param_name} in range {param_values} ({len(param_values)} values)')

results_all = []

for whisker_reward in [True, False]:
    print(f"Running RL model with whisker_reward={whisker_reward}")

    for param_val in param_values:

    #wh_color = 'forestgreen' if whisker_reward else 'crimson'
        #p_aud, p_whisk, p_ns, Q_exc_hist, Q_inh_hist = rl_model_proba(Nmax=Nmax, dt=dt, whisker_reward=whisker_reward,
        #                                                              **{param_name: param_val})
        p_aud, p_whisk, p_ns, Q_exc_hist, Q_inh_hist = rl_model_proba(Nmax=Nmax, dt=dt, whisker_reward=whisker_reward)

        # Save results
        results = {}
        results['whisker_reward'] = [whisker_reward for _ in range(len(p_aud))]
        results['time'] = np.arange(len(p_aud)) * dt
        results['p_aud'] = p_aud
        results['p_whisk'] = p_whisk
        results['p_ns'] = p_ns
        #results['Q_exc_hist'] = Q_exc_hist
        #results['Q_inh_hist'] = Q_inh_hist
        results['Q_ea'] = Q_exc_hist[:, 0]  # Auditory Q
        results['Q_ew'] = Q_exc_hist[:, 1]  # Whisker Q
        results['Q_en'] = Q_exc_hist[:, 2]  # No Stim Q
        results['Q_ec'] = Q_exc_hist[:, 3]  # C Q
        results['Q_ia'] = Q_inh_hist[:, 0]  # Auditory Inhibitory Q
        results['Q_iw'] = Q_inh_hist[:, 1]  # Whisker Inhibitory Q
        results['Q_in'] = Q_inh_hist[:, 2]  # No Stim Inhibitory Q
        results['Q_ic'] = Q_inh_hist[:, 3]  # C Inhibitory Q
        results[param_name] = [param_val for _ in range(len(p_aud))]
        results = pd.DataFrame(results)
        results_all.append(results)

    do_plot=False
    if do_plot:
        # Plot results
        t = np.arange(len(p_aud)) * dt
        wh_color = 'forestgreen' if whisker_reward else 'crimson'

        # Plot performance
        plt.figure(figsize=(10, 5))
        plt.plot(t, p_aud, label="Auditory", color='mediumblue')
        plt.plot(t, p_whisk, label="Whisker", color=wh_color)
        plt.plot(t, p_ns, label="No stimulus", color='k')
        #plt.plot(t, prob_correct, label="Avg. Correct", color='black')
        plt.xlabel("Trial")
        plt.ylabel("P(lick)")
        plt.ylim(0, 1.05)
        plt.legend(loc='center right', frameon=False)
        plt.tight_layout()
        #plt.show()

        # Plot synaptic weights
        fig, axs = plt.subplots(1,2, figsize=(12, 5), sharey=False)
        Q_exc_hist = np.array(Q_exc_hist)
        Q_inh_hist = np.array(Q_inh_hist)
        axs[0].plot(t, Q_exc_hist[:, 0], label='Q_exc A', color='mediumblue')
        axs[0].plot(t, Q_exc_hist[:, 1], label='Q_exc W', color=wh_color)
        axs[0].plot(t, Q_exc_hist[:, 2], label='Q_exc NS', color='k')
        axs[0].plot(t, Q_exc_hist[:, 3], label='Q_exc C', color='grey')
        axs[1].plot(t, Q_inh_hist[:, 0], label='Q_inh A', color='mediumblue')
        axs[1].plot(t, Q_inh_hist[:, 1], label='Q_inh W', color=wh_color)
        axs[1].plot(t, Q_inh_hist[:, 2], label='Q_inh NS', color='k')
        axs[1].plot(t, Q_inh_hist[:, 3], label='Q_inh C', color='grey')
        for ax in axs.flat:
            ax.set_xlabel("Trial")
            ax.set_ylabel("Synaptic weight")
            ax.legend(loc='center right', frameon=False)
        fig.tight_layout()
        #plt.show()


results_df = pd.concat(results_all)

# Plot combined results
fig, ax = plt.subplots(figsize=(5, 3), dpi=400)

show_ahr = True
if show_ahr:
    if param_name == 'Q_ea':
        results_df_group = results_df[results_df['whisker_reward'] == True]
        #results_df['Q_ea'] = results_df['Q_ea'].astype('category')
        #n_colors = results_df['Q_ea'].nunique()
        #print(n_colors)

        sns.lineplot(data=results_df[results_df.whisker_reward == 1],
                     ax=ax,
                     x='time',
                     y='p_aud',
                     hue='Q_ea',
                     #hue_order=param_values,
                     #palette=putils.make_cmap_n_from_color_lite2dark('mediumblue', n_colors=len(param_values)),
                     #palette=sns.color_palette('Purples', n_colors=len(param_values)),
                     palette=sns.light_palette('mediumblue', reverse=False, n_colors=len(param_values), as_cmap=False),
                     lw=1.5,
                     legend=True,
                     )
        #sns.lineplot(data=results_df[results_df.whisker_reward == 0],
        #             ax=ax,
        #             x='time',
        #             y='p_aud',
        #             hue='Q_ea',
        #             #hue_order=param_values,
        #             #palette=putils.make_cmap_n_from_color_lite2dark('lightblue', n_colors=len(param_values)),
        #             #palette=sns.color_palette('Blues', n_colors=len(param_values)),
        #             palette=sns.light_palette('lightblue', reverse=False, n_colors=len(param_values), as_cmap=False),
        #
        #             lw=1.5,
        #             legend=False,
        #             )
    else:
        sns.lineplot(data=results_df,
                     ax=ax,
                     x='time',
                     y='p_aud',
                     hue='whisker_reward',
                     hue_order=[True,False],
                     palette=['mediumblue', 'lightblue'],
                     lw=2,
                     legend=False,
                     )

show_whr = True
if show_whr:
    # Plot whisker performance
    if param_name == 'Q_ew':
        sns.lineplot(data=results_df[results_df.whisker_reward == True],
                     ax=ax,
                     x='time',
                     y='p_whisk',
                     hue='Q_ew',
                     palette=sns.light_palette('forestgreen', reverse=False, n_colors=len(param_values), as_cmap=False),
                     lw=1.5,
                     legend=True
                     )
        sns.lineplot(data=results_df[results_df.whisker_reward == False],
                     ax=ax,
                     x='time',
                     y='p_whisk',
                     hue='Q_ew',
                     palette=sns.light_palette('crimson', reverse=False, n_colors=len(param_values), as_cmap=False),
                     lw=1.5,
                     legend=True
                     )
    else:
        sns.lineplot(data=results_df,
                     ax=ax,
                     x='time',
                     y='p_whisk',
                     hue='whisker_reward',
                     hue_order=[True,False],
                     palette=['forestgreen', 'crimson'],
                     lw=2,
                     legend=False
                     )
    if param_name == 'wh_reward_proba':
        sns.lineplot(data=results_df[results_df.whisker_reward==True],
                     ax=ax,
                     x='time',
                     y='p_whisk',
                     hue='wh_reward_proba',
                     palette='Greens',
                     lw=2,
                     legend=False
                     )
# Plot false alarm performance
show_far = True
if show_far:
    if param_name == 'Q_in':
        results_df_group = results_df[results_df.whisker_reward == True]
        sns.lineplot(data=results_df_group,
                     ax=ax,
                     x='time',
                     y='p_ns',
                     hue='Q_in',
                     #hue_order=[True,False],
                     palette=sns.light_palette('k', reverse=False, n_colors=len(param_values), as_cmap=False),
                     lw=1.5,
                     legend=True
                     )
    else:
        sns.lineplot(data=results_df,
                     ax=ax,
                     x='time',
                     y='p_ns',
                     hue='whisker_reward',
                     hue_order=[True,False],
                     palette=['k', 'grey'],
                     lw=2,
                     legend=False
                     )

ax.set_xlabel("Trials")
ax.set_ylabel("P(lick)")
ax.set_ylim(0, 1.05)
ax.legend(loc='center right', frameon=False, fontsize=8)
fig.tight_layout()
plt.show()
