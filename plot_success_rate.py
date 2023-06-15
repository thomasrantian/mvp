import numpy as np
import matplotlib.pyplot as plt


ground_truth_success_rate_20_env_1_seed = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/6_6_ground_gruth/0f5dc14c-7efe-457e-88d7-4ac7390d7488/success_rate.npy")
OT_success_rate_20_env_1_seed = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/6_6_ot/9140fdaf-ad2c-4257-9958-05900432d450/success_rate.npy")
ground_truth_success_rate_24_env_8_seed = np.load("/home/thomastian/workspace/mvp_exp_data/success_rate_24_env_8_seed.npy")



# plot the success rate with legend
plt.plot(ground_truth_success_rate_20_env_1_seed, label="Ground Truth, 20env, 1 seed")
plt.plot(OT_success_rate_20_env_1_seed, label="Preference, 20env, 1 seed")
plt.plot(ground_truth_success_rate_24_env_8_seed, label="Ground Truth, 24env, 8 seed")
plt.xlabel("Iteration")
plt.ylabel("Success Rate")
plt.legend()
plt.show()