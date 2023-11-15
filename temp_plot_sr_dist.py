import numpy as np
import matplotlib.pyplot as plt


def plot_sr(exp_1_path, exp_2_path):
    sr_exp1 = np.load(exp_1_path)
    sr_exp2 = np.load(exp_2_path)

    sr_mean = np.mean([sr_exp1, sr_exp2], axis=0)
    sr_std = np.std([sr_exp1, sr_exp2], axis=0)

    plt.plot(sr_mean)
    plt.fill_between(np.arange(len(sr_mean)), sr_mean-sr_std, sr_mean+sr_std, alpha=0.2)
    plt.show()

base_dir = '/home/thomastian/workspace/mvp_exp_data/rl_runs/'



# Franka RLHF
exp_1_path = base_dir + '9_13_RLHF_Sig_franka_exp/150/9e79e1f4-180f-4286-bbdf-da1810cf0412/success_rate.npy'
exp_2_path = base_dir + '9_14_paper_results_RLHF_franka/300/8a5e66a4-f3af-4a1f-be8b-d7f5768f88bc/success_rate.npy'
plot_sr(exp_1_path, exp_2_path)



# Kuka ours
# exp_1_path = base_dir + '9_16_paper_results_OT_kuka/150/e5a86508-ad8a-42ca-8099-3ee29d81f73f/success_rate.npy'
# exp_2_path = base_dir + '9_12_OT_Kuka_datasize_exp/150/491b5e1f-864c-4b2e-8645-56345664bb85/success_rate.npy'
# plot_sr(exp_1_path, exp_2_path)
