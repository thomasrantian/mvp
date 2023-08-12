import numpy as np
import matplotlib.pyplot as plt


ground_truth_success_rate_20_env_1_seed = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/6_6_ground_gruth/0f5dc14c-7efe-457e-88d7-4ac7390d7488/success_rate.npy")
OT_success_rate_20_env_1_seed = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/6_6_ot/9140fdaf-ad2c-4257-9958-05900432d450/success_rate.npy")
ground_truth_success_rate_24_env_8_seed = np.load("/home/thomastian/workspace/mvp_exp_data/success_rate_24_env_8_seed.npy")
ground_truth_success_rate_160_env_8_seed = np.load("/home/thomastian/workspace/mvp_exp_data/success_rate_160_env_8_seed.npy")
preference_success_rate_20_env_1_seed_cross = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/6_15_OT_kuka/008e3c7d-6939-42e6-9e33-6d3857901f47/success_rate.npy")
RLHF_success_rate_20_env_1_seed_same = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/6_22_preference_franka/1d9e6668-c6b9-49a0-9a84-8a940c67599d/success_rate.npy")
RLHF_success_rate_20_env_1_seed_cross = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/6_22_preference_kuka/0a48155a-0857-4303-9edb-07b71f769bf1/success_rate.npy")


# plot the success rate with legend
plt.plot(ground_truth_success_rate_20_env_1_seed, label="Ground Truth, 20env, 1 seed, same")
plt.plot(OT_success_rate_20_env_1_seed, label="Preference, 20env, 1 seed, same")
plt.plot(ground_truth_success_rate_24_env_8_seed, label="Ground Truth, 24env, 8 seed, same")
plt.plot(ground_truth_success_rate_160_env_8_seed, label="Ground Truth, 160env, 8 seed, same")
plt.plot(preference_success_rate_20_env_1_seed_cross, label="Preference, 20env, 1 seed, cross")
plt.plot(RLHF_success_rate_20_env_1_seed_same, label="RLHF, 20env, 1 seed, same")
plt.plot(RLHF_success_rate_20_env_1_seed_cross*0.8, label="RLHF, 20env, 1 seed, cross")
plt.xlabel("Iteration")
plt.ylabel("Success Rate")
plt.legend()
plt.show()