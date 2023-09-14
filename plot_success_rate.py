import numpy as np
import matplotlib.pyplot as plt


#ground_truth_success_rate_20_env_1_seed = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/6_6_ground_gruth/0f5dc14c-7efe-457e-88d7-4ac7390d7488/success_rate.npy")
OT_success_rate_20_env_1_seed = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/6_6_ot/9140fdaf-ad2c-4257-9958-05900432d450/success_rate.npy")
#ground_truth_success_rate_24_env_8_seed = np.load("/home/thomastian/workspace/mvp_exp_data/success_rate_24_env_8_seed.npy")
#ground_truth_success_rate_160_env_8_seed = np.load("/home/thomastian/workspace/mvp_exp_data/success_rate_160_env_8_seed.npy")
preference_success_rate_20_env_1_seed_cross = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/6_15_OT_kuka/008e3c7d-6939-42e6-9e33-6d3857901f47/success_rate.npy")
RLHF_success_rate_20_env_1_seed_same = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/6_22_preference_franka/1d9e6668-c6b9-49a0-9a84-8a940c67599d/success_rate.npy")
#RLHF_success_rate_20_env_1_seed_cross = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/6_22_preference_kuka/0a48155a-0857-4303-9edb-07b71f769bf1/success_rate.npy")

data_150 = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/9_9_OT_franka_datasize_exp/150/fcb252da-2049-40c0-85ac-034d73d6b7f6/success_rate.npy")
data_100 = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/8_31_OT_franka_datasize_exp/100/7e82954c-e730-4d2b-9fc2-9de2d1f3c565/success_rate.npy")
data_50 = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/8_31_OT_franka_datasize_exp/50/638d3802-fe5c-402c-9c76-59dabe2e361b/success_rate.npy")

data_RLHF_150 = np.load('/home/thomastian/workspace/mvp_exp_data/rl_runs/9_9_RLHF_franka_datasize_exp/150/ae1a8ed3-5e09-4d91-9dc0-579dadc81d6f/success_rate.npy')

data_RLHF_150_old = np.load('/home/thomastian/workspace/mvp_exp_data/rl_runs/9_12_RLHF_franka_datasize_exp/150/3d1cbcbe-e0e9-415b-a1bb-ed1b5bda57d8/success_rate.npy')


OT_data_150_lr1e3 = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/9_12_OT_franka_datasize_exp/150/16312627-c990-450c-b5e4-ec853a31cdce/success_rate.npy")

OT_data_150_kuka = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/9_12_OT_Kuka_datasize_exp/150/491b5e1f-864c-4b2e-8645-56345664bb85/success_rate.npy")

RLHF_data_150_kuka = np.load("/home/thomastian/workspace/mvp_exp_data/rl_runs/9_12_RLHF_Kuka_datasize_exp/150/5fef19ae-c008-420f-8c3d-9e0c54317244/success_rate.npy")


# plot the success rate with legend
#plt.plot(ground_truth_success_rate_20_env_1_seed, label="Ground Truth, 20env, 1 seed, same")
plt.plot(OT_success_rate_20_env_1_seed, label="Preference, 20env, 1 seed, same")
#plt.plot(ground_truth_success_rate_24_env_8_seed, label="Ground Truth, 24env, 8 seed, same")
#plt.plot(ground_truth_success_rate_160_env_8_seed, label="Ground Truth, 160env, 8 seed, same")
plt.plot(preference_success_rate_20_env_1_seed_cross, label="Preference, 20env, 1 seed, cross")
#plt.plot(RLHF_success_rate_20_env_1_seed_same, label="RLHF, 20env, 1 seed, same")
#plt.plot(RLHF_success_rate_20_env_1_seed_cross*0.8, label="RLHF, 20env, 1 seed, cross")
plt.plot(data_150, label="150")
plt.plot(data_100, label="100")
plt.plot(data_50, label="50")
plt.plot(data_RLHF_150, label="RLHF 150")
plt.plot(data_RLHF_150_old,'-.', label="RLHF 150 old")
plt.plot(OT_data_150_lr1e3, label="OT 150 lr1e-3")
plt.plot(OT_data_150_kuka, ':',label = 'OT_data_150_kuka')
plt.plot(RLHF_data_150_kuka, label = 'RLHF_data_150_kuka')
plt.xlabel("Iteration")
plt.ylabel("Success Rate")
plt.legend()
plt.show()