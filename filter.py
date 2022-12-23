import numpy as np
import math
from job_generator import create_workload

##Simple Discretization function
def Discretize(step,val):
    if(val==0):
        return 0
    else:
        return math.ceil((1/step)*val)

## Mathematical way to find minimum speed but only applies for preemption
## It is not used in our model (could need some adjustments to double check its validity)
def get_min_speed_preemption(workload):       ## Works assuming preemption
    HighPrioWorkload = workload[np.where(workload[:,3] == 1),:][0]
    print("HIGH PRIO JOBS")
    print(HighPrioWorkload)
    print("-------------")
    print("SUM OF WCET : ", np.sum(HighPrioWorkload[:,2]))
    print('--')
    print("MAX DEADLINE : ", np.max(HighPrioWorkload[:,1]))
    print('--')
    print("MIN RELEASE TIME : ", np.min(HighPrioWorkload[:,0]))
    for i,v in enumerate(HighPrioWorkload):
        if(i != len(HighPrioWorkload)):
            curr_workload = HighPrioWorkload[i:i+2]
            print(curr_workload)
            s = np.round(np.sum(curr_workload[:,2]) / (np.max(curr_workload[:,1]) - np.min(curr_workload[:,0]) ),2)
            print(s)
    min_s = np.round(np.sum(HighPrioWorkload[:,2]) / (np.max(HighPrioWorkload[:,1]) - np.min(HighPrioWorkload[:,0]) ),2)
    print("Miniumum Speed affordable by this schedule: ")
    return min_s


## This function is used to run a previously generated workload but with a specific speed
## The purpose is to empirically find the min_speed for the given workload parameter
def check_speed(speed,workload,HiPrioLen):
    for row in workload:
        row[4] = 0
        row[5] = 0
    for row in workload:
        row[2] = math.ceil(row[2] / speed)


    idx=np.argmin(workload[:,0])                                           ##index of first job released
    workload[idx,4]=1                                                      ## Mark Jobs as Executed
    t = workload[idx,0] + workload[idx,2]
    new_array=workload[idx,:]                                              ##New array of jobs that are scheduled properly

    while(len(workload) >= 1):
        workload[np.where((workload[:,2]+t) > workload[:,1]),5]=1              ##Mark all jobs that are starved (Their WCET is not enough to complete)
        # workload[np.where(workload[:,1] < t) ,5] = 1                           ##Mark all jobs with deadlines that already passed
        workload = np.delete(workload, np.where(workload[:,4]==1), axis=0)     ##Delete Executed Jobs
        workload = np.delete(workload, np.where(workload[:,5]==1), axis=0)     ##Delete Starved jobs
        if (len(workload) >= 1):
            curr_min_index = -1
            curr_min_val = 1e12
            for i,row in enumerate(workload):
                if(row[1] < curr_min_val and row[0] <= t):
                    curr_min_val = row[1]
                    curr_min_index = i

            if(curr_min_index != -1):
                toSchedule = curr_min_index
                workload[toSchedule,4] = 1
                t += np.ceil(workload[toSchedule,2])
                new_array = np.concatenate((new_array,workload[toSchedule,:]))
            else:
                t+= np.ceil(workload[np.argmin(workload[:,0]),0])

    new_array.resize((len(new_array)//6,6))
    new_array=new_array[:,:6]
    # print("HI PRIO : ", HiPrioLen)
    # print("neew arr :", len(new_array))

    if(HiPrioLen == len(new_array)):
        return True
    else:
        return False

## This function includes Discretization and non-preemptive EDF Filtration
## It also adds the extra features (remaining time and laxity)
def filter_dataset(num_jobs, total_load, lo_per, job_density):

    workload = np.zeros((num_jobs, 6))
    workload[:, :4] = create_workload(num_jobs, total_load, lo_per, job_density)

    scaling_step = 0.1
    for row in workload:
        row[0] = (math.modf(row[0])[1]*(1/scaling_step)) + Discretize(scaling_step, math.modf(row[0])[0])
        row[1] = (math.modf(row[1])[1]*(1/scaling_step)) + Discretize(scaling_step, math.modf(row[1])[0])
        row[2] = (math.modf(row[2])[1]*(1/scaling_step)) + Discretize(scaling_step, math.modf(row[2])[0])

    t = 0
    idx=np.argmin(workload[:,0])                                           ##index of job released at  t=0
    workload[idx,4]=1                                                      ## Mark Jobs as Executed
    t += workload[idx,2]                                                   ## Move time step to end of chosen job (No preemption)
    new_array=workload[idx,:]                                              ##New array of jobs that are scheduled properly

    # print("HERE 1")
    while(len(workload) >= 1):
        workload[np.where((workload[:,2]+t) > workload[:,1]),5]=1              ##Mark all jobs that are starved (Their WCET is not enough to complete)
        # workload[np.where(workload[:,1] < t) ,5] = 1                           ##Mark all jobs with deadlines that already passed
        workload = np.delete(workload, np.where(workload[:,4]==1), axis=0)     ##Delete Executed Jobs
        workload = np.delete(workload, np.where(workload[:,5]==1), axis=0)     ##Delete Starved jobs

        if (len(workload) >=1):
            curr_min_index = -1
            curr_min_val = 1e12
            for i,row in enumerate(workload):
                if(row[1] < curr_min_val and row[0] <= t):
                    curr_min_val = row[1]
                    curr_min_index = i

            if(curr_min_index != -1):
                toSchedule = curr_min_index
                workload[toSchedule,4] = 1
                t += workload[toSchedule,2]
                new_array = np.concatenate((new_array,workload[toSchedule,:]))
            else:
                t+= workload[np.argmin(workload[:,0]),0]

    new_array.resize((len(new_array)//6,6))
    new_array=new_array[:,:6]
    for i,v in enumerate(new_array):
        new_array[i][4] = v[2] ## Remaining Time
        new_array[i][5] = v[1] - v[4] ## Laxity
    # print(len(new_array))
    return new_array


## This function is called to generate the set of jobs J with its respective minimum speed
def generateData(total_load,lo_per,job_density,job_num,threshold):

    # print("Generating Workload")
    workload_filtered = filter_dataset(job_num, total_load, lo_per,job_density)
    # print("Generated Workload")
    while(True):
        curr_min_speed = 2
        if len(workload_filtered) >= job_num*threshold:
            HiWorkload = workload_filtered[np.where(workload_filtered[:,3] == 1),:][0]
            HiPrioLen = len(HiWorkload)
            if(HiPrioLen > 0):
                for speed in np.arange(1,0.05,-0.05):
                    speed = np.round(speed,2)
                    if(check_speed(speed,HiWorkload,HiPrioLen)):
                        if(speed < curr_min_speed):
                            curr_min_speed = speed
        if(curr_min_speed <= 1):
            break
        workload_filtered = filter_dataset(job_num, total_load, lo_per,job_density)
    # print("Generated done with speed")
    return workload_filtered,curr_min_speed