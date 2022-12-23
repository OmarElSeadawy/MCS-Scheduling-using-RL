import ray
import pickle
from environment_generator import MCOnlineNewEnv,MCOnlineLatestTestEnv,MCOnlineTestEnv,MCOnlineLatestTestEnvPre
# from ray.rllib.agents.dqn import ApexTrainer
from ray.tune.registry import register_env
from ray.rllib.algorithms.apex_dqn import ApexDQN as ApexTrainer
from ray import tune
import numpy as np
from gym.spaces import Box, MultiBinary, Discrete, Dict
import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.dqn as dqn
import ray.rllib.algorithms.dreamer as dreamer
import ray.rllib.algorithms.marwil as marwil
import ray.rllib.algorithms.simple_q as simpleq
import ray.rllib.algorithms.ars as ars
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)



# PPO_Checkpoint = '/home/omarelseadawy/ray_results/PPO_MCOnlinePPO100_2022-11-12_02-18-20tlxes_b9/checkpoint_000451/'
# DQN_Checkpoint = '/home/omarelseadawy/ray_results/DQN_MCOnlineDQNOG100Jobs_2022-11-12_02-47-137qroehzg/checkpoint_000451/'
# SIQ_Checkpoint = '/home/omarelseadawy/ray_results/SimpleQ_MCOnlineSimpleQ100_2022-11-12_03-01-42r9mymt5w/checkpoint_000451/'
# ARS_Checkpoint = '/home/omarelseadawy/ray_results/ARS_MCOnlineARS100_2022-11-12_10-04-38vdfvkho1/checkpoint_001451/'
# APX_Checkpoint = '/home/omarelseadawy/ray_results/ApexDQN_2022-11-12_11-00-43/ApexDQN_MCOnlineCustom_7d687_00000_0_2022-11-12_11-00-44/checkpoint_000250/'
MAR_Checkpoint = '/home/omarelseadawy/ray_results/MARWIL_MCOnlineMARWIL100Jobs_2022-11-17_21-13-279s840s4m/checkpoint_000451/'

# APX_Checkpoint = '/home/omarelseadawy/ray_results/ApexDQN_2022-12-10_16-14-37/ApexDQN_MCOnlineTestNoDisc_faa69_00000_0_2022-12-10_16-14-37/checkpoint_000050/'
# APX_Checkpoint = '/home/omarelseadawy/ray_results/ApexDQN_2022-12-10_20-07-38/ApexDQN_MCOnlinePreempt_8811b_00000_0_2022-12-10_20-07-38/checkpoint_000050/'

## PREEMPTION EVAL
# APX_Checkpoint = '/home/omarelseadawy/ray_results/ApexDQNImportantPreempt/ApexDQN_MCOnlineNewRewardPreempt2_9ce0e_00000_0_2022-12-12_23-54-55/checkpoint_000050/'
## NONPREEMPEVAL
# APX_Checkpoint = '/home/omarelseadawy/ray_results/ApexDQNImportantNonPreempt3/ApexDQN_MCOnlineNewRewardNonPreempt3_bb3ba_00000_0_2022-12-12_19-02-16/checkpoint_000050/'


ALGORITHM = "NewMARRun"

ray.init()

# agent = ApexTrainer(config={"env": "OnlineEvaluatinoPPO", "num_workers": 3, "num_gpus": 0})
# policy = agent.workers.local_worker().get_policy()


#
# print("Setting Up PPO Env")
# register_env("OnlineEvaluationPPO", lambda env_config: MCOnlineLatestTestEnvPre())
# ppoconfig = ppo.DEFAULT_CONFIG.copy()
# ppoenv = MCOnlineLatestTestEnvPre()
# ppoconfig["num_gpus"]=0
# ppoconfig["num_workers"]=4
# ppoconfig["env"]="OnlineEvaluationPPO"
# ppoagent = ppo.PPO(config=ppoconfig)
# ppoagent.restore(PPO_Checkpoint)
# ppo_episode_reward = 0
# ppodone = False


# print("Setting Up DQN Env")
# register_env("OnlineEvaluationDQN", lambda env_config: MCOnlineLatestTestEnvPre())
# dqnconfig = dqn.DEFAULT_CONFIG.copy()
# dqnenv = MCOnlineLatestTestEnv()
# dqnconfig["num_gpus"]=0
# dqnconfig["num_workers"]=4
# dqnconfig["env"]="OnlineEvaluationDQN"
# dqnagent = dqn.DQN(config=dqnconfig)
# dqnagent.restore(DQN_Checkpoint)
# # print("Starting Env")
# env = MCOnlineLatestTestEnvPre()
# state = env.reset()
#
# print("Setting Up simpleq Env")
# register_env("OnlineEvaluationsimpleq", lambda env_config: MCOnlineLatestTestEnv())
# simpleqconfig = simpleq.DEFAULT_CONFIG.copy()
# simpleqenv = MCOnlineLatestTestEnv()
# simpleqconfig["num_gpus"]=0
# simpleqconfig["num_workers"]=4
# simpleqconfig["env"]="OnlineEvaluationsimpleq"
# simpleqagent = simpleq.SimpleQ(config=simpleqconfig)
# simpleqagent.restore(SIQ_Checkpoint)
# # print("Starting Env")
# env = MCOnlineLatestTestEnv()
# state = env.reset()


print("Setting Up MARWIL Env")
register_env("OnlineEvaluationMAR", lambda env_config: MCOnlineLatestTestEnvPre())
marwilconfig = marwil.DEFAULT_CONFIG.copy()
marwilenv = MCOnlineLatestTestEnvPre()
marwilconfig["num_gpus"]=0
marwilconfig["num_workers"]=4
marwilconfig["env"]="OnlineEvaluationMAR"
marwilagent = marwil.MARWIL(config=marwilconfig)
marwilagent.restore(MAR_Checkpoint)
print("Starting Env")
# env = MCOnlineLatestTestEnv()
env = MCOnlineLatestTestEnvPre()
state = env.reset()
#
# print("Setting Up ARS Env")
# register_env("OnlineEvaluationsARS", lambda env_config: MCOnlineLatestTestEnvPre())
# arsconfig = ars.DEFAULT_CONFIG.copy()
# arsenv = MCOnlineLatestTestEnvPre()
# arsconfig["num_gpus"]=0
# arsconfig["num_workers"]=4
# arsconfig["env"]="OnlineEvaluationsARS"
# arsagent = ars.ARS(config=arsconfig)
# arsagent.restore(ARS_Checkpoint)
# print("Starting Env")
# env = MCOnlineLatestTestEnv()
# state = env.reset()

# env = MCOnlineLatestTestEnvPre()
# state = env.reset()

# print("Setting Up CustomApex Env")
# # env = MCOnlineLatestTestEnv()
# env = MCOnlineLatestTestEnvPre()
# register_env("OnlineEvaluationsAPXNOPRE", lambda env_config: env)
# apxagent = ApexTrainer(config={"env": "OnlineEvaluationsAPXNOPRE", "num_workers": 3, "num_gpus": 0})
# apxagent.restore(APX_Checkpoint)


# print("Starting PPO Env")
# ppoenv = env
# ppostate = ppoenv.reset()
print("Start Experiment")
# episode_reward = 0
# done = False
# i = 0
# while not ppodone:
    # ppoaction = ppoagent.compute_single_action(ppostate)
    # ppoobs,pporeward,ppodone,ppoinfo = ppoenv.step(ppoaction)
    # ppo_episode_reward += pporeward

# print("Final Reward : ", ppo_episode_reward)


## Path for Testing Datasets
testdatapath = 'newpervsnonhunderd/dataset'
for datasetno in range(9):
    print("Current Dataset No. " + str(datasetno))
    with open(testdatapath+str(datasetno)+'metadata.pickle', 'rb') as metafile:
        metadata = pickle.load(metafile)
    print("Total Load : ", metadata[0])
    print("Lo Percentage : ", metadata[1])
    print("Job Density : ", metadata[2])

    ## Loading Dataset metadata from pickled file and writing to results file
    with open("Results/"+ALGORITHM+"results"+str(datasetno)+".txt", "w+") as resultsfile:
        resultsfile.write("Total Load : " + str(metadata[0]) + '\n')
        resultsfile.write("Lo Percentage : " + str(metadata[1]) + '\n')
        resultsfile.write("Job Density : " + str(metadata[2]) + '\n')

    ## Variables in monitoring and analysing results
    accum = []
    NumberofJobs = 50
    Experiments = 2000
    Xaxis = 5

    HC_jobs_completed = np.zeros((Experiments, Xaxis))
    HC_jobs = np.zeros((Experiments, Xaxis))
    LC_jobs_completed = np.zeros((Experiments, Xaxis))
    LC_jobs = np.zeros((Experiments, Xaxis))
    total_jobs_completed = np.zeros((Experiments, Xaxis))
    total_jobs = np.zeros((Experiments, Xaxis))
    time_wasted = np.zeros((Experiments, Xaxis))

    workload_filtered_arr = []
    min_speed_arr = []
    jobnumber_arr = []
    ## Loading Dataset workload instances from pickled file for testing
    with open(testdatapath + str(datasetno) + '.pickle', 'rb') as datafile:
        while 1:
            try:
                workloadinstance = pickle.load(datafile)
                workload_filtered_arr.append(workloadinstance[0])
                min_speed_arr.append(workloadinstance[1])
                jobnumber_arr.append(workloadinstance[2])
            except EOFError:
                break
    # datafile.close()
    print("LOADED DATA SET SUCCESSFULLY")
    print(len(workload_filtered_arr))
    # print(workload_filtered_arr)
    print(min_speed_arr[0])
    print(jobnumber_arr[0])
    for i in range(Xaxis):
        print(i)
        degradation_threshold = np.round(0.2 + 0.2 * i, 2)
        print("Experiment AT DEGRADATION CHANCE = ", degradation_threshold)
        for j in range(Experiments):
            if(j % 100 == 0):
                print("EXPERIMENT NO: ", j)
            workload_filtered = workload_filtered_arr[j]
            min_speed = min_speed_arr[j]
            jobnumber = jobnumber_arr[j]

            state = env.reset()
            env.degradation_threshold = degradation_threshold
            env.lowest_speed = min_speed

            newworkload = np.zeros((NumberofJobs, 9))
            newworkload[:workload_filtered.size // 6, :6] = workload_filtered
            newworkload[workload_filtered.size // 6:, [0, 2, 3]] = 0
            newworkload[workload_filtered.size // 6:, 1] = np.max(workload_filtered[:, 1])
            newworkload[workload_filtered.size // 6:, 8] = 1
            newworkload = np.abs(newworkload)
            env.workload = newworkload
            env.updateBuffer()
            env.degradationHappens()

            total_jobs[j, i] = len(env.workload)

            # action = ppoagent.compute_single_action(state)
            # action = dqnagent.compute_single_action(state)
            # action = simpleqagent.compute_single_action(state)
            # action = arsagent.compute_single_action(state)
            # action = apxagent.compute_single_action(state)
            action = marwilagent.compute_single_action(state)
            state, reward, done, info = env.step(action)
            # episode_reward += reward

            # action = ppoagent.compute_actions([state])
            # state, reward, done, info = env.step(action[0][0])
#
            # counter = 1000                  ## This counter was used as fail-safe to make sure the agent didn't aimlessly run forever in case of a mistake
            while (info['done'] == False):
                # action = ppoagent.compute_single_action(state)
                # action = dqnagent.compute_single_action(state)
                # action = simpleqagent.compute_single_action(state)
                # action = arsagent.compute_single_action(state)
                # action = apxagent.compute_single_action(state)
                action = marwilagent.compute_single_action(state)
                state, reward, done, info = env.step(action)

                # if (counter == 0):
                #     break
                # else:
                #     counter -= 1
#
            for row in env.workload:
                if (row[3] == 1):
                    HC_jobs[j, i] += 1
                if (row[3] == 0):
                    LC_jobs[j, i] += 1
                if (row[3] == 1 and row[8] == 1):
                    HC_jobs_completed[j, i] += 1
                if (row[3] == 0 and row[8] == 1):
                    LC_jobs_completed[j, i] += 1
                if (row[8] == 1):
                    total_jobs_completed[j, i] += 1
                if (row[4] < 0):
                    time_wasted[j, i] += row[4]
#
    ## Calculating Average values for experiments for one dataset

    ##Average Time Wasted over all experiments
    time_wasted_completed = np.array((np.mean(time_wasted, axis=0)))
    print("TimeWasted ", time_wasted_completed)


    ##Average Total Jobs and Total completed jobs over all experiments
    total_jobs = np.array((np.mean(total_jobs, axis=0)))
    print("total_jobs", total_jobs)
    total_jobs_completed = np.array((np.mean(total_jobs_completed, axis=0)))
    print("TotCompleted ", total_jobs_completed)


    ##Average HI jobs and Completed HI jobs over all experiments
    HC_jobs = np.array((np.mean(HC_jobs, axis=0)))
    print("Total HC : ", HC_jobs)
    HC_jobs_completed = np.array((np.mean(HC_jobs_completed, axis=0)))
    print("Hi2 ", HC_jobs_completed)
    HC_jobs_percent = np.array((HC_jobs_completed/HC_jobs))
    print("HiPerc", HC_jobs_percent)

    ##Average LO jobs and Completed LO jobs over all experiments
    LC_jobs = np.array((np.mean(LC_jobs, axis=0)))
    print("Total LC : ", LC_jobs)
    LC_jobs_completed = np.array((np.mean(LC_jobs_completed, axis=0)))
    print("lo2", LC_jobs_completed)
    LC_jobs_percent = np.array((LC_jobs_completed / LC_jobs))
    print("LoPerc", LC_jobs_percent)

    ## Write the results in resultsfile to avoid losing them or in case of quick re-evaluation
    with open("Results/"+ALGORITHM+"results" + str(datasetno) + ".txt", "a+") as resultsfile:
        resultsfile.write("\nTime Wasted Completed\n")
        resultsfile.write(str(time_wasted_completed))
        resultsfile.write('\n')

        resultsfile.write("\nTotal Jobs \n")
        resultsfile.write(str(total_jobs))
        resultsfile.write('\n')

        resultsfile.write("\nTotal Jobs Completed\n")
        resultsfile.write(str(total_jobs_completed))
        resultsfile.write('\n')

        resultsfile.write("\nTotal Hi Criticality Jobs\n")
        resultsfile.write(str(HC_jobs))
        resultsfile.write('\n')

        resultsfile.write("\nTotal Hi Criticality Jobs Completed\n")
        resultsfile.write(str(HC_jobs_completed))
        resultsfile.write('\n')

        resultsfile.write("\nTotal Low Criticality Jobs \n")
        resultsfile.write(str(LC_jobs))
        resultsfile.write('\n')

        resultsfile.write("\nTotal Low Criticality Jobs Completed\n")
        resultsfile.write(str(LC_jobs_completed))
        resultsfile.write('\n')

        resultsfile.write("\nTotal Hi Perc Jobs Completed\n")
        resultsfile.write(str(HC_jobs_percent))
        resultsfile.write('\n')

        resultsfile.write("\nTotal Lo Perc Jobs Completed\n")
        resultsfile.write(str(LC_jobs_percent))
        resultsfile.write('\n')


    # PLOTTING the graphs
    degrade_threshold_arr = []
    for i in range(5):
        degrade_threshold_arr.append(np.round(0.2 + (i * 0.2), 2))
    print(degrade_threshold_arr)

    plt.plot(degrade_threshold_arr, time_wasted_completed, 'black', label='Total Time Wasted')
    plt.xlabel(ALGORITHM+"Degradation Percentage Threshold")
    plt.title(ALGORITHM+"Online Non-Preemption Time Wasting Evaluation")
    plt.savefig("Results/"+ALGORITHM+"online_timewaste_dataset"+str(datasetno)+".png")
    plt.clf()

    plt.plot(degrade_threshold_arr, HC_jobs, 'black', label='Total Number of HC Jobs')
    plt.plot(degrade_threshold_arr, HC_jobs_completed, 'red', label='Hi-critical Jobs Completed')
    plt.xlabel(ALGORITHM+"Degradation Percentage Threshold")
    plt.ylim(0, NumberofJobs + 1)
    plt.title(ALGORITHM+"Number of Complete HC Jobs")
    plt.savefig("Results/"+ALGORITHM+"Online_hc_jobs_completed_dataset"+str(datasetno)+".png")
    plt.clf()

    plt.plot(degrade_threshold_arr, LC_jobs, 'black', label='Total Number of LC Jobs')
    plt.plot(degrade_threshold_arr, LC_jobs_completed, 'yellow', label='Lo-criticality Jobs Completed')
    plt.xlabel(ALGORITHM+"Degradation Percentage Threshold")
    plt.ylim(0, NumberofJobs + 1)
    plt.title(ALGORITHM+"Number of Complete LC Jobs")
    plt.savefig("Results/"+ALGORITHM+"Online_lc_jobs_completed_dataset"+str(datasetno)+".png")
    plt.clf()

    plt.plot(degrade_threshold_arr, total_jobs_completed, 'black', label='Total Jobs Completed')
    plt.plot(degrade_threshold_arr, HC_jobs_completed, 'red', label='Hi-critical Jobs Completed')
    plt.plot(degrade_threshold_arr, LC_jobs_completed, 'green', label='Low-criticalJobsCompleted')
    plt.plot(degrade_threshold_arr, total_jobs, 'yellow', label='Total Jobs')
    plt.ylabel(ALGORITHM+"Percentage of Jobs Completed")
    plt.xlabel("Degradation Percentage Threshold")
    plt.ylim(0, NumberofJobs + 1)
    plt.title(ALGORITHM+"Online Non-Preemption Degradation Evaluation")
    plt.savefig("Results/"+ALGORITHM+"Online-nonpreemption_dataset"+str(datasetno)+".png")
    plt.clf()
    # plt.show()