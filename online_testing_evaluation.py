import ray
import pickle
from environment_generator import MCOnlineNewEnv,MCOnlineTestEnv
from ray.rllib.agents.dqn import ApexTrainer
from ray.tune.registry import register_env
import numpy as np
import matplotlib.pyplot as plt


checkpoint_path = '{path}/ray_results/APEX_2021-12-01_22-04-17/APEX_MCOnlineNewEnvNewAdjustment_dcfeb_00000_0_2021-12-01_22-04-17/checkpoint_000740/checkpoint-740'

ray.init()
## The reason we use MCOnlineTestEnv and not the MCOnlineNewEnv used for training
## Is because we want to feed a specific dataset to the model
## Rest of the environments are exactly identical
env = MCOnlineTestEnv()

register_env("OnlineEval", lambda env_config: env)
agent = ApexTrainer(config={"env": "OnlineEval", "num_workers": 12, "num_gpus": 1})

## Restore policy from agent
agent.restore(checkpoint_path)
policy = agent.workers.local_worker().get_policy()

## Path for Testing Datasets
testdatapath = 'NewTestingDataset10k/dataset'
for datasetno in range(9):
    print("Current Dataset No. " + str(datasetno))
    metafile = open(testdatapath+str(datasetno)+'metadata.pickle', 'rb')
    metadata = pickle.load(metafile)
    print("Total Load : ", metadata[0])
    print("Lo Percentage : ", metadata[1])
    print("Job Density : ", metadata[2])

    ## Loading Dataset metadata from pickled file and writing to results file
    with open("Results/results"+str(datasetno)+".txt", "w+") as resultsfile:
        resultsfile.write("Total Load : " + str(metadata[0]) + '\n')
        resultsfile.write("Lo Percentage : " + str(metadata[1]) + '\n')
        resultsfile.write("Job Density : " + str(metadata[2]) + '\n')

    ## Variables in monitoring and analysing results
    accum = []
    NumberofJobs = 30
    Experiments = 10000
    Xaxis = 20

    HC_jobs_completed = np.zeros((Experiments, Xaxis))
    HC_jobs = np.zeros((Experiments, Xaxis))
    LC_jobs_completed = np.zeros((Experiments, Xaxis))
    LC_jobs = np.zeros((Experiments, Xaxis))
    total_jobs_completed = np.zeros((Experiments, Xaxis))
    total_jobs = np.zeros((Experiments, Xaxis))
    time_wasted = np.zeros((Experiments, Xaxis))

    ## Loading Dataset workload instances from pickled file for testing
    datafile = open(testdatapath + str(datasetno) + '.pickle', 'rb')
    workload_filtered_arr = []
    min_speed_arr = []
    jobnumber_arr = []
    while 1:
        try:
            workloadinstance = pickle.load(datafile)
            workload_filtered_arr.append(workloadinstance[0])
            min_speed_arr.append(workloadinstance[1])
            jobnumber_arr.append(workloadinstance[2])
        except EOFError:
            break
    datafile.close()
    print("LOADED DATA SET SUCCESSFULLY")
    print(workload_filtered_arr[0])
    print(min_speed_arr[0])
    print(jobnumber_arr[0])
    for i in range(Xaxis):
        degradation_threshold = np.round(0.05 + 0.05 * i, 2)
        print("Experiment AT DEGRADATION CHANCE = ", degradation_threshold)
        for j in range(Experiments):
            workload_filtered = workload_filtered_arr[j]
            min_speed = min_speed_arr[j]
            jobnumber = jobnumber_arr[j]

            state = env.reset()
            env.degradation_threshold = degradation_threshold
            env.lowest_speed = min_speed

            newworkload = np.zeros((jobnumber, 9))
            newworkload[:workload_filtered.size // 6, :6] = workload_filtered
            newworkload[workload_filtered.size // 6:, [0, 2, 3]] = 0
            newworkload[workload_filtered.size // 6:, 1] = np.max(workload_filtered[:, 1])
            newworkload[workload_filtered.size // 6:, 8] = 1
            newworkload = np.abs(newworkload)
            env.workload = newworkload
            env.updateBuffer()
            env.degradationHappens()

            total_jobs[j, i] = len(env.workload)
            action = policy.compute_actions([state])
            state, reward, done, info = env.step(action[0][0])

            # counter = 1000                  ## This counter was used as fail-safe to make sure the agent didn't aimlessly run forever in case of a mistake
            while (info['done'] == False):
                action = policy.compute_actions([state])
                state, reward, done, info = env.step(action[0][0])
                # if (counter == 0):
                #     break
                # else:
                #     counter -= 1

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


    ##Average LO jobs and Completed LO jobs over all experiments
    LC_jobs = np.array((np.mean(LC_jobs, axis=0)))
    print("Total LC : ", LC_jobs)
    LC_jobs_completed = np.array((np.mean(LC_jobs_completed, axis=0)))
    print("lo2", LC_jobs_completed)


    ## Write the results in resultsfile to avoid losing them or in case of quick re-evaluation
    with open("Results/results" + str(datasetno) + ".txt", "a+") as resultsfile:
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


    # PLOTTING the graphs
    degrade_threshold_arr = []
    for i in range(20):
        degrade_threshold_arr.append(np.round(0.05 + (i * 0.05), 2))
    print(degrade_threshold_arr)

    plt.plot(degrade_threshold_arr, time_wasted_completed, 'black', label='Total Time Wasted')
    plt.xlabel("Degradation Percentage Threshold")
    plt.title("Online Non-Preemption Time Wasting Evaluation")
    plt.savefig("Results/online_timewaste_dataset"+str(datasetno)+".png")
    plt.clf()

    plt.plot(degrade_threshold_arr, HC_jobs, 'black', label='Total Number of HC Jobs')
    plt.plot(degrade_threshold_arr, HC_jobs_completed, 'red', label='Hi-critical Jobs Completed')
    plt.xlabel("Degradation Percentage Threshold")
    plt.ylim(0, NumberofJobs + 1)
    plt.title("Number of Complete HC Jobs")
    plt.savefig("Results/Online_hc_jobs_completed_dataset"+str(datasetno)+".png")
    plt.clf()

    plt.plot(degrade_threshold_arr, LC_jobs, 'black', label='Total Number of LC Jobs')
    plt.plot(degrade_threshold_arr, LC_jobs_completed, 'yellow', label='Lo-criticality Jobs Completed')
    plt.xlabel("Degradation Percentage Threshold")
    plt.ylim(0, NumberofJobs + 1)
    plt.title("Number of Complete LC Jobs")
    plt.savefig("Results/Online_lc_jobs_completed_dataset"+str(datasetno)+".png")
    plt.clf()

    plt.plot(degrade_threshold_arr, total_jobs_completed, 'black', label='Total Jobs Completed')
    plt.plot(degrade_threshold_arr, HC_jobs_completed, 'red', label='Hi-critical Jobs Completed')
    plt.plot(degrade_threshold_arr, LC_jobs_completed, 'green', label='Low-criticalJobsCompleted')
    plt.plot(degrade_threshold_arr, total_jobs, 'yellow', label='Total Jobs')
    plt.ylabel("Percentage of Jobs Completed")
    plt.xlabel("Degradation Percentage Threshold")
    plt.ylim(0, NumberofJobs + 1)
    plt.title("Online Non-Preemption Degradation Evaluation")
    plt.savefig("Results/Online-nonpreemption_dataset"+str(datasetno)+".png")
    plt.show()