# ## This evaluation was inteded for Offline Environment Evaluation

# import gym, ray
# from environment_generator import MCOfflineNewEnv,MCOnlineNewEnv
# from ray.rllib.agents.dqn import ApexTrainer
# from ray.tune.registry import register_env
# import numpy as np
# import matplotlib.pyplot as plt
# np.set_printoptions(suppress=True)

# ## Random Checkpoint location from trained model
# checkpoint_path = '{path}/ray_results/APEX_2021-11-18_15-56-28/APEX_Test2021-11-18_15-56-28/checkpoint_000010/checkpoint-10'


# ray.init()
# env = MCOfflineNewEnv()
# register_env("EvaluationEnv", lambda env_config: env)
# agent = ApexTrainer(config={"env": "EvaluationEnv", "num_workers": 4, "num_gpus": 1})

# ## Restor Policy from agent
# agent.restore(checkpoint_path)
# policy = agent.workers.local_worker().get_policy()


# ## All variables for monitoring purposes
# accum = []
# timehorizon = 35000000
# NumberofJobs = 30
# Experiments = 1
# Xaxis = 1
# HC_jobs_completed = np.zeros((Experiments, Xaxis))
# HC_jobs = np.zeros((Experiments, Xaxis))
# LC_jobs_completed = np.zeros((Experiments, Xaxis))
# LC_jobs = np.zeros((Experiments,Xaxis))
# total_jobs_completed = np.zeros((Experiments, Xaxis))
# total_jobs = np.zeros((Experiments, Xaxis))
# time_wasted = np.zeros((Experiments, Xaxis))
# CPUProcessing = np.ones((Xaxis, timehorizon))

# for i in range(Xaxis):
#     degradation_threshold = 0.05 + 0.05*i
#     for j in range(Experiments):
#         state = env.reset()
#         env.degradation_threshold = degradation_threshold
#         total_jobs[j, i] = len(env.workload)
#         action = policy.compute_actions([state])
#         state, reward, done, info = env.step(action[0][0])

#         ## Info is a dictionary that is updated with every step
#         ## It is very important for debugging purposes
#         ## It has values for workload instances, actions chosen, rewards and other important info
#         # print(info.keys())

#         counter = NumberofJobs + 5
#         while(info['done'] == False):
#             action = policy.compute_actions([state])
#             state, reward, done, info = env.step(action[0][0])
#             if(counter == 0):
#                 break
#             else:
#                 counter -= 1

#         for row in env.workload:
#             if(row[3] == 1):
#                 HC_jobs[j,i] += 1
#             if(row[3] == 0):
#                 LC_jobs[j,i] += 1
#             if (row[3] == 1 and row[8] == 1):
#                 HC_jobs_completed[j,i] += 1
#             if (row[3] == 0 and row[8] == 1):
#                 LC_jobs_completed[j,i] += 1
#             if (row[8] == 1):
#                 total_jobs_completed[j,i] += 1
#             if(row[4] < 0):
#                 time_wasted[j,i] += row[4]


#     ## Degradation Modelling (Very slow to plot for large time horizons)
#     #     for element in env.degradationschedule:
#     #         cputime = int(element[0])
#     #         degradationsp = element[1]
#     #         currentsp = CPUProcessing[i][cputime]
#     #         CPUProcessing[i][cputime] = (degradationsp + currentsp) / 2
#     # plt.plot(list(range(timehorizon)), CPUProcessing[i], 'black', label='CPU Speed')
#     # plt.title("Mean CPU Performance")
#     # plt.ylim(0.0, 1.0)
#     # plt.savefig(str(np.round(degradation_threshold,2))+"%ThresholdPerformance.png")
#     # plt.clf()


# # for i in range(len(CPUProcessing)):
# #     print(CPUProcessing[i])

# ## Average time wasted over all experiments for each parameter
# time_wasted_completed=np.array((np.mean(time_wasted,axis=0)))
# print("TimeWasted ",time_wasted_completed)

# ## Average total jobs and total completed over all experiments for each parameter
# total_jobs = np.array((np.mean(total_jobs,axis=0)))
# print("total_jobs",total_jobs)
# total_jobs_completed=np.array((np.mean(total_jobs_completed,axis=0)))
# print("TotCompleted ",total_jobs_completed)

# ## Average Number of HI jobs and completed HI Jobs over all experiments for each parameter
# HC_jobs = np.array((np.mean(HC_jobs,axis=0)))
# print("Total HC : ", HC_jobs)
# HC_jobs_completed = np.array((np.mean(HC_jobs_completed,axis=0)))
# print("Hi2 ",HC_jobs_completed)

# ## Average Number of LO jobs and completed LO Jobs over all experiments for each parameter
# LC_jobs = np.array((np.mean(LC_jobs,axis=0)))
# print("Total LC : ", LC_jobs)
# LC_jobs_completed = np.array((np.mean(LC_jobs_completed,axis=0)))
# print("lo2",LC_jobs_completed)



# # PLOTTING over degradation threshold
# degrade_threshold_arr = []
# for i in range(20):
#     degrade_threshold_arr.append(np.round(0.05+(i*0.05),2))
# print(degrade_threshold_arr)

# plt.plot(degrade_threshold_arr,time_wasted_completed,'black', label='Total Time Wasted')
# plt.xlabel("Degradation Percentage Threshold")
# plt.title("Offline Non-Preemption Time Wasting Evaluation")
# plt.savefig("offline_timewaste.png")
# plt.clf()


# plt.plot(degrade_threshold_arr, HC_jobs, 'black', label='Total Number of HC Jobs')
# plt.plot(degrade_threshold_arr, HC_jobs_completed, 'red', label='Hi-critical Jobs Completed')
# plt.xlabel("Degradation Percentage Threshold")
# plt.ylim(0, NumberofJobs+1)
# plt.title("Number of Complete HC Jobs")
# plt.savefig("offline_hc_jobs_completed.png")
# plt.clf()


# plt.plot(degrade_threshold_arr, LC_jobs, 'black', label='Total Number of LC Jobs')
# plt.plot(degrade_threshold_arr, LC_jobs_completed, 'yellow', label='Lo-criticality Jobs Completed')
# plt.xlabel("Degradation Percentage Threshold")
# plt.ylim(0, NumberofJobs+1)
# plt.title("Number of Complete LC Jobs")
# plt.savefig("offline_lc_jobs_completed.png")
# plt.clf()


# plt.plot(degrade_threshold_arr, total_jobs_completed, 'black', label='Total Jobs Completed')
# plt.plot(degrade_threshold_arr, HC_jobs_completed, 'red', label='Hi-critical Jobs Completed')
# plt.plot(degrade_threshold_arr, LC_jobs_completed, 'green', label='Low-criticalJobsCompleted')
# plt.plot(degrade_threshold_arr, total_jobs, 'yellow', label='Total Jobs')
# plt.ylabel("Percentage of Jobs Completed")
# plt.xlabel("Degradation Percentage Threshold")
# plt.ylim(0, NumberofJobs+1)
# plt.title("Offline Non-Preemption Degradation Evaluation")
# plt.savefig("offline-nonpreemption.png")
# plt.show()