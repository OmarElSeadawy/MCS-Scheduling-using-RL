#!/usr/bin/env python
# coding: utf-8

import math
import pickle
import numpy as np
import gym
from gym.utils import seeding
from gym.spaces import Box, MultiBinary, Discrete, Dict
from filter import generateData
from gym.spaces.utils import flatten, flatten_space
import random

class MCOfflineNewEnv(gym.Env):
    def __init__(self, env_config= {'job_num': 20, 'total_load': 0.4, 'lo_per': 0.3, 'job_density': 4}):

        self.time = 0                     ## Time Horizon for the episode
        self.seed()
        self.speed = 1
        self.filter_threshold = 0.9       ## Allows data generator to drop 10% of data for faster processing and generation


        ## Set of Variables for the job generator (Parameters defined are advised)
        self.job_num = env_config['job_num']
        self.degradation_threshold = np.random.uniform(low=0.05,high=1)
        self.total_load = np.random.uniform(low=0.2, high=1)#env_config['total_load']
        self.lo_per = np.random.uniform(low=0, high=1-2/self.job_num) #env_config['lo_per']
        self.job_density = np.random.randint(low=int(self.job_num*(1/4)), high=int(self.job_num*(1/2)))##env_config['job_density']

        workload_filtered,lowest_speed = generateData(self.total_load,self.lo_per,self.job_density,self.job_num,self.filter_threshold)
        self.workload_filtered = workload_filtered
        self.lowest_speed = lowest_speed
        ## workload is the creation of the job list from data generation
        ## The 9 columns stand for (ReleaseTime,Deadline,WCET,Criticality,RemainingTime,AdjustedPriority,ReleaseStatus,StarvationStatus,ExecutionStatus)
        # The first 6 are generated from the dataset filtration and the rest are status parameters


        ## Dummy definitions that add jobs in case some were dropped because of leniency threshold
        workload = np.zeros((self.job_num, 9))
        workload[:workload_filtered.size // 6, :6] = workload_filtered
        workload[workload_filtered.size // 6:, [0, 2, 3]] = 0
        workload[workload_filtered.size // 6:, 1] = np.max(workload_filtered[:, 1])
        workload[workload_filtered.size // 6:, 8] = 1

        self.workload = np.abs(workload)

        self.action_space = Discrete(self.job_num)          ##Discrete action space not continuous
        self.observation_space_dict = Dict({
            'action_mask': Box(0, 1, shape=(self.job_num,)),
            'avail_actions': Box(-np.inf, np.inf, shape=(self.job_num,)),
            'MCenv': Dict({
                'RDP_jobs': Box(low=0, high=np.inf, shape=(self.job_num, 3)),
                'C_jobs': MultiBinary(self.job_num * 1),
                'RA_jobs': Box(low=-10, high=np.inf, shape=(self.job_num, 2)),
                'ReleasedParameter': MultiBinary(self.job_num),
                'StarvedParameter': MultiBinary(self.job_num),
                'ExecutedParameter': MultiBinary(self.job_num),
                'ProcessorSpeed':Box(low=np.array([0.]), high=np.array([np.inf])),
            })
        })
        self.observation_space = flatten_space(self.observation_space_dict)

        self.degradationschedule = []

        self.workload[:, 6][self.time >= self.workload[:, 0]] = 1     ##Check jobs that are released ( t > rj) making their release status = 1
        self.workload[:, 7][self.time + self.workload[:, 2]/self.speed > self.workload[:, 1]] = 1  #Expired Jobs (their processing time > deadline) , label starvation status = 1

        ## Action assignment
        self.action_mask = np.ones(self.job_num)

        self.action_assignments = np.ones(self.job_num)
        # self.action_assignments = np.zeros_like(self.workload[:, :6])      ##Marks all non-released jobs as zero
        # self.action_assignments[self.action_mask.astype(bool)] = self.workload[self.action_mask.astype(bool), :6]
        self._update_available()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def degradationHappens(self):
        if(random.random() < self.degradation_threshold):
            return True
        else:
            return False

    def step(self, action):
        self.done = self._done()     ## Terminate if Done
        prev_workload = np.copy(self.workload)
        self.reward = 0
        self.info["timebefore"] = self.time
        self.info["workload_before"] = self.workload
        self.info["done"] = self.done
        ## Negative Reward if agent picked an executed or starved job
        if self.workload[action, 7] == 1 or self.workload[action, 8] == 1:
            self._update_available()
            self.state = self._get_obs()
            self.reward = -10
            self.done = self._done()
            return self.state, self.reward, self.done, self.info

        self.time = max(self.time, self.workload[action, 0])

        time_unit = 1
        deadline_flag = False
        degraded_period = 0

        self.info["degradation_speed"] = 0
        self.info["total_degradation"] = 0

        while(self.workload[action,4] > 0):
            if(self.time >= self.workload[action,1]):          ## Deadline is met and job hasn't finished
                deadline_flag = True
                break

            if(self.degradationHappens()):
                degraded_speed = np.random.uniform(low=self.lowest_speed, high=1)
                self.workload[action, 4] -= degraded_speed
                self.degradationschedule.append((self.time, degraded_speed))
            else:
                self.workload[action, 4] -= 1

            self.time += 1

        if (deadline_flag == False):
            self.workload[action, 8] = 1      ##label job as executed
            if(self.workload[action, 3] == 1):
                self.reward += 2
            else:
                self.reward += 1

            self.done = self._done()
        else:   ## If a job was chosen and it failed to meet deadline
            self.workload[action, 7] = 1
            self.reward = -1

        self.workload[:, 6][self.time >= self.workload[:, 0]] = 1        ##update currently released jobs
        expired_jobs = (self.time >= self.workload[:, 1]) * (1-self.workload[:, 8]).astype(bool)      ##Already Expired
        self.workload[:, 7][expired_jobs] = 1
        starving_jobs = (self.time+self.workload[:,2] > self.workload[:,1]) * (1-self.workload[:, 8]).astype(bool) ## Already starved
        self.workload[:, 7][starving_jobs] = 1
        self._update_available()
        self.state = self._get_obs()
        self.done = self._done()
        self.info["done"]= self.done
        self.info["reward"]= self.reward
        self.info["current_action"] = self.workload[action, :]
        self.info["workload_after"] = self.workload
        self.info["timeafter"] = self.time
        return self.state, self.reward, self.done, self.info

    def _update_available(self):
        self.action_mask[self.workload[:, 7].astype(bool) | self.workload[:, 8].astype(bool)] = 0
        self.action_assignments = np.zeros_like(self.workload[:, :6])
        self.action_assignments[self.action_mask.astype(bool)] = self.workload[self.action_mask.astype(bool), :6]

    def _get_obs(self):
        obs_dict = dict({
            'action_mask': self.action_mask,
            'avail_actions': self.action_assignments,
            'MCenv': dict({
                'RDP_jobs': np.array(self.workload[:, :3]),  ## Release Deadline Processing Jobs
                'C_jobs': np.array(self.workload[:, 3]).flatten(),  ## Criticality Column
                'RA_jobs': np.array(self.workload[:, 4:6]),  ## Remaining time and Adjusted Priority
                'ReleasedParameter': np.array(self.workload[:, 6]).flatten(),  ## Rest of parameters
                'StarvedParameter': np.array(self.workload[:, 7]).flatten(),
                'ExecutedParameter': np.array(self.workload[:, 8]).flatten(),
                'ProcessorSpeed': np.array([self.speed]).flatten()
            })
        })
        return flatten(self.observation_space_dict, obs_dict)

    def reset(self):
        self.time = 0                     ## Time Horizon for the episode
        self.seed()
        self.speed = 1
        self.degradation_threshold = np.random.uniform(low=0.05,high=0.95)
        self.filter_threshold = 0.9

        self.total_load = np.random.uniform(low=0.2, high=1)#env_config['total_load']
        self.lo_per = np.random.uniform(low=0, high=1-2/self.job_num) #env_config['lo_per']
        self.job_density = np.random.randint(low=int(self.job_num*(1/4)), high=int(self.job_num*(1/2)))##env_config['job_density']
        self.filter_threshold = 1

        workload_filtered,lowest_speed = generateData(self.total_load,self.lo_per,self.job_density,self.job_num,self.filter_threshold)
        self.lowest_speed=lowest_speed
        self.workload_filtered = workload_filtered

        workload = np.zeros((self.job_num, 9))
        workload[:workload_filtered.size // 6, :6] = workload_filtered
        workload[workload_filtered.size // 6:, [0, 2, 3]] = 0
        workload[workload_filtered.size // 6:, 1] = np.max(workload_filtered[:, 1])
        workload[workload_filtered.size // 6:, 8] = 1

        self.workload = np.abs(workload)

        self.action_space = Discrete(self.job_num)          ##Discrete action space not continuous

        self.workload[:, 6][self.time >= self.workload[:, 0]] = 1     ##Check jobs that are released ( t > rj) making their release status = 1
        self.workload[:, 7][self.time + self.workload[:, 2]/self.speed > self.workload[:, 1]] = 1  #Expired Jobs (their processing time > deadline) , label starvation status = 1


        self.degradationschedule = []

        self.done = False
        self.info = {}
        self.reward = 0
        self.state = self._get_obs()

        ## Action assignment
        self.action_mask = np.ones(self.job_num)

        self.action_assignments = np.ones(self.job_num)
        # self.action_assignments = np.zeros_like(self.workload[:, :6])      ##Marks all non-released jobs as zero
        # self.action_assignments[self.action_mask.astype(bool)] = self.workload[self.action_mask.astype(bool), :6]
        self._update_available()
        return self.state

    def _done(self):
        arr = []
        for row in self.workload: ## Done when all jobs are executed or starved
            if(row[7] == 1 or row[8] == 1):
                arr.append(0)
        # max = -1
        # for row in self.workload:
        #     if(row[1] > max):
        #         max = row[1]

        if(len(arr) == len(self.workload)):
            return True
        # elif(self.time >= max):
        #     return True
        else:
            return False
        # return bool((self.workload[:, 7].astype(bool) | self.workload[:, 8].astype(bool)).all())

class MCOnlineNewEnv(gym.Env):
    def __init__(self, env_config= {'total_load': 0.4, 'lo_per': 0.3, 'job_density': 4, 'buffer_length': 5}):

        self.seed()                   ## Random Seed
        self.time = 0                 ## Initializing Time Horizon for the episode
        self.speed = 1                ## Initializing Speed = 1
        self.degradation_threshold = np.random.uniform(low=0.05, high=0.95)
        self.filter_threshold = 0.9
        self.buffer_length = env_config['buffer_length']                                        ## Online Buffer Length
        # self.job_num = np.random.randint(low=1 * self.buffer_length, high=2 * self.buffer_length)                    ## Number of Jobs based on random int
        self.job_num = 30
        self.total_load = np.random.uniform(low=0.2, high=1)                                    # env_config['total_load']
        self.lo_per = np.random.uniform(low=0, high=1 - 2 / self.job_num)                       # env_config['lo_per']
        self.job_density = np.random.randint(low=self.job_num*(1/4), high=self.job_num*(1/2))   # env_config['job_density']


        ## workload is the creation of the job list from data generation
        ## The 9 columns stand for (ReleaseTime,Deadline,WCET,Criticality,RemainingTime,AdjustedPriority,ReleaseStatus,StarvationStatus,ExecutionStatus)
        # The first 6 are generated from the dataset filtration and the rest are status parameters
        workload_filtered, lowest_speed = generateData(self.total_load, self.lo_per, self.job_density, self.job_num,
                                                       self.filter_threshold)
        self.workload_filtered = workload_filtered
        self.lowest_speed = lowest_speed

        #Dumping generated training data for analysis
        traindataf = open('traindata.pickle', 'ab')
        pickle.dump((self.workload_filtered, self.lowest_speed, self.degradation_threshold,self.total_load, self.lo_per, self.job_density),traindataf)

        workload = np.zeros((self.job_num, 9))

        workload[:workload_filtered.size // 6, :6] = workload_filtered
        workload[workload_filtered.size // 6:, [0, 2, 3]] = 0
        workload[workload_filtered.size // 6:, 1] = np.max(workload_filtered[:, 1])
        workload[workload_filtered.size // 6:, 8] = 1

        self.workload = np.abs(self.workload)


        ## Action Space for the agent
        self.action_space = Discrete(self.buffer_length)

        ## Observation Space
        self.observation_space_dict = Dict({
            'action_mask': Box(0, 1, shape=(self.buffer_length,)),
            'avail_actions': Box(-np.inf, np.inf, shape=(self.buffer_length,)),
            'MCenv': Dict({
                'Online_Buffer': Box(low=-2, high=np.inf ,shape=(self.buffer_length,)),
                'C_jobs': MultiBinary(self.buffer_length),
                'RemLaxity_jobs': Box(low=-np.inf, high=np.inf, shape=(self.buffer_length, 2)),
                'ProcessorSpeed':Box(low=np.array([0.]), high=np.array([np.inf])),
            })
        })

        self.observation_space= flatten_space(self.observation_space_dict)

        self.degradationschedule = []

        self.workbuffer = np.zeros((self.buffer_length, self.workload.shape[1]))
        for row in self.workbuffer:      ## Making Deadline infinite for dummy jobs
            row[1] = np.inf

        self.online_buffer = [-1] * self.buffer_length
        self.updateBuffer()


        ## Action assignment
        self.action_mask = np.ones(self.buffer_length)

        self.action_assignments = np.ones(self.buffer_length)


        self._update_available()

    def updateBuffer(self):
        self._update_workload()
        self.current_released_jobs = 0
        self.online_buffer = [-1]*self.buffer_length
        self.workload = self.workload[self.workload[:, 5].argsort()]
        counter = 0
        for idx, val in enumerate(self.workload):
            if(val[6] == 1 and val[7] != 1 and val[8] != 1):
                self.online_buffer[counter] = idx
                counter += 1
            if(counter >= self.buffer_length):
                break
        self.current_released_jobs = counter

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def degradationHappens(self):
        if(random.random() < self.degradation_threshold):
            self.speed = np.random.uniform(low=self.lowest_speed, high=1)
        else:
            self.speed = 1

    def step(self, action):
        self.done = self._done()  ## Terminate if Done
        prev_workload = np.copy(self.workload)
        self.reward = 0
        self.info["timebefore"] = self.time
        self.info["workload_before"] = self.workload
        self.info["onlinebuffer_before"] = self.online_buffer
        self.info["done"] = self.done

        if self.current_released_jobs == 0:
            self.time += 1
            self.updateBuffer()
            self._update_available()
            self.state = self._get_obs()
            self.reward = 0
            self.done = self._done()
            return self.state, self.reward, self.done, self.info

        current_action = self.online_buffer[action]
        self.info["currently_released_jobs"] = self.current_released_jobs
        self.info["onlinebuffer_withjobs"] = self.online_buffer
        self.info["ACTION_INDEX"] = current_action
        self.info["current_action"] = self.workload[current_action, :]

        if current_action == -1:
            self._update_available()
            self.state = self._get_obs()
            self.reward = -25
            self.done = self._done()
            return self.state, self.reward, self.done, self.info
        else:
            degradaded_action = True
            if self.speed == 1:
                degradaded_action = False

            if self.workload[current_action, 7] == 1 or self.workload[current_action, 8] == 1:
                self._update_available()
                self.state = self._get_obs()
                self.reward = -10
                self.done = self._done()
                return self.state, self.reward, self.done, self.info

            self.time = max(self.time, self.workload[current_action, 0])

            low_jobs_count = 0
            hi_jobs_count = 0
            for val in self.online_buffer:
                if val != -1:
                    if self.workload[current_action,3] == 1:
                        hi_jobs_count += 1
                    else:
                        low_jobs_count += 1

            ## Inverse Rank of Laxity in online buffer
            laxity_rank = 0
            for idx,val in enumerate(self.online_buffer):
                if val == current_action and val != 1:
                    laxity_rank = self.buffer_length - idx
                    break

            deadline_flag = False
            while (self.workload[current_action, 4] > 0):
                if (self.time >= self.workload[current_action, 1]):  ## Deadline is met and job hasn't finished
                    deadline_flag = True
                    break

                self.workload[current_action, 4] -= self.speed
                if(self.speed < 1):
                    self.degradationschedule.append((self.time, self.speed))

                self.time += 1
                self.degradationHappens()

            if(deadline_flag == False):
                self.workload[current_action, 8] = 1 ## Mark job as executed
                #### CALCULATE REWARD ####
                if self.current_released_jobs <= 1:
                    self.reward = 0
                else:
                    if degradaded_action == False:     ## Action was taken under normal speed
                        self.reward += laxity_rank
                    else:                              ## Action was taken under degraded speed
                        if low_jobs_count == 0:
                            self.reward += laxity_rank
                        elif hi_jobs_count == 0:
                            self.reward += laxity_rank
                        elif self.workload[current_action, 3] == 1:
                            self.reward += self.buffer_length + laxity_rank
                        else:
                            self.reward -= 1

                ## Upadting Laxity of jobs
                for row in self.workload:
                    row[5] = row[1] - row[4]
                self.workload[:, 6][self.time >= self.workload[:, 0]] = 1  ##update currently released jobs
                expired_jobs = (self.time >= self.workload[:, 1]) * (1 - self.workload[:, 8]).astype(bool)  ##Already Expired
                self.workload[:, 7][expired_jobs] = 1
                starving_jobs = (self.time + self.workload[:, 2] > self.workload[:, 1]) * (1 - self.workload[:, 8]).astype(bool)  ## Already starved
                self.workload[:, 7][starving_jobs] = 1
                self.done = self._done()
            else:  ## If a job was chosen and it failed to meet deadline
                self.workload[current_action, 7] = 1
                self.reward = -1

        self.updateBuffer()
        self._update_available()
        self.state = self._get_obs()
        self.done = self._done()
        self.info["done"] = self.done
        self.info["reward"] = self.reward
        self.info["onlinebuffer_after"] = self.online_buffer
        self.info["ACTION_INDEX"] = current_action
        self.info["current_action"] = self.workload[current_action, :]
        self.info["workload_after"] = self.workload
        self.info["timeafter"] = self.time
        return self.state, self.reward, self.done, self.info

    def _update_available(self):
        self.degradationHappens()
        for idx, val in enumerate(self.online_buffer):
            if(val == -1):
                self.action_mask[idx] = 0

    def _update_workload(self):
        ## Upadting Laxity of jobs
        for row in self.workload:
            row[5] = row[1] - row[4]
        ## Updating Released jobs to be available and expiring the starved jobs
        self.workload[:, 6][self.time >= self.workload[:, 0]] = 1     ##Check jobs that are released ( t > rj) making their release status = 1
        self.workload[:, 7][self.time + self.workload[:, 2] > self.workload[:, 1]] = 1  #Expired Jobs (their processing time > deadline) , label starvation status = 1

    def _get_obs(self):
        try:
            self.workbuffer = np.zeros((self.buffer_length, self.workload.shape[1]))
            for row in self.workbuffer:      ## Making Deadline infinite for dummy jobs
                row[1] = np.inf
            counter = 0
            for row in self.online_buffer:
                if row != -1:
                    self.workbuffer[counter] = self.workload[row]
                    counter += 1
                if counter >= self.buffer_length:
                    break
        except:
            print("ERROR with work buffer")

        obs_dict = dict({
            'action_mask': self.action_mask,
            'avail_actions': self.action_assignments,
            'MCenv': dict({
                'Online_Buffer': np.array(self.online_buffer),
                'C_jobs': np.array(self.workbuffer[:, 3]).flatten(),            ## Criticality Column
                'RemLaxity_jobs': np.array(self.workbuffer[:, 4:6]),            ## Remaining time and Adjusted Priority
                'ProcessorSpeed': np.array([self.speed]).flatten()
            })
        })
        return flatten(self.observation_space_dict, obs_dict)

    def reset(self):

        self.time = 0
        self.seed()
        self.speed = 1
        self.degradation_threshold = np.random.uniform(low=0.05, high=0.95)
        self.filter_threshold = 0.9

        self.job_num = 30
        self.total_load = np.random.uniform(low=0.2, high=1)  # env_config['total_load']
        self.lo_per = np.random.uniform(low=0, high=1 - 2 / self.job_num)  # env_config['lo_per']
        self.job_density = np.random.randint(low=self.job_num*(1/4), high=self.job_num*(1/2))  # env_config['job_density']

        ## workload is the creation of the job list from data generation
        ## The 9 columns stand for (ReleaseTime,Deadline,WCET,Criticality,RemainingTime,AdjustedPriority,ReleaseStatus,StarvationStatus,ExecutionStatus)
        # The first 6 are generated from the dataset filtration and the rest are status parameters
        workload_filtered, lowest_speed = generateData(self.total_load, self.lo_per, self.job_density, self.job_num,
                                                       self.filter_threshold)
        self.lowest_speed = lowest_speed
        self.workload_filtered = workload_filtered
        traindataf = open('traindata.pickle','ab')
        pickle.dump((self.workload_filtered,self.lowest_speed,self.degradation_threshold,self.total_load,self.lo_per,self.job_density),traindataf)
        self.workload_filtered = workload_filtered# self.job_num = workload_filtered.shape[0]
        workload = np.zeros((self.job_num, 9))

        # print("FILTERED WORKLAOD")
        # print(workload_filtered)
        # print(workload_filtered.size)
        workload[:workload_filtered.size // 6, :6] = workload_filtered
        workload[workload_filtered.size // 6:, [0, 2, 3]] = 0
        workload[workload_filtered.size // 6:, 1] = np.max(workload_filtered[:, 1])
        workload[workload_filtered.size // 6:, 8] = 1


        self.workload = np.abs(workload)

        self.workload[:, 6][self.time >= self.workload[:, 0]] = 1  ##Check jobs that are released ( t > rj) making their release status = 1
        self.workload[:, 7][self.time + self.workload[:, 2] / self.speed > self.workload[:, 1]] = 1  # Expired Jobs (their processing time > deadline) , label starvation status = 1

        self.done = False
        self.info = {}
        self.reward = 0
        self._update_available()
        self.state = self._get_obs()
        self.degradationschedule = []

        self.workbuffer = np.zeros((self.buffer_length, self.workload.shape[1]))
        for row in self.workbuffer:      ## Making Deadline infinite for dummy jobs
            row[1] = np.inf
        self.online_buffer = [-1] * self.buffer_length

        self.updateBuffer()

        self.action_mask = np.ones(self.buffer_length)

        self.action_assignments = np.ones(self.buffer_length)

        self._update_available()
        return self.state

    def _done(self):
        arr = []
        for row in self.workload:
            if (row[7] == 1 or row[8] == 1):
                arr.append(0)
        if (len(arr) == len(self.workload)):
            return True
        else:
            return False



## THIS ENVIRONMENT IS ONLY USED FOR TESTING
## THE ONLY DIFFERENCE IS THAT IT DOESNT USE GENERATED WORKLOADS BUT RATHER GETS WORKLOAD FROM EVALUATION SCRIPT
## EVERYTHING ELSE IS EXACTLY THE SAME

class MCOnlineTestEnv(gym.Env):
    def __init__(self, env_config= {'total_load': 0.4, 'lo_per': 0.3, 'job_density': 4, 'buffer_length': 5}):

        self.seed()                   ## Random Seed
        self.time = 0                 ## Initializing Time Horizon for the episode
        self.speed = 1                ## Initializing Speed = 1
        self.degradation_threshold = np.random.uniform(low=0.05, high=0.95)
        self.filter_threshold = 0.9
        self.buffer_length = env_config['buffer_length']                                        ## Online Buffer Length
        # self.job_num = np.random.randint(low=1 * self.buffer_length, high=2 * self.buffer_length)                    ## Number of Jobs based on random int
        self.job_num = 30
        self.total_load = np.random.uniform(low=0.2, high=1)                                    # env_config['total_load']
        self.lo_per = np.random.uniform(low=0, high=1 - 2 / self.job_num)                       # env_config['lo_per']
        self.job_density = np.random.randint(low=self.job_num*(1/4), high=self.job_num*(1/2))   # env_config['job_density']


        self.workload = np.zeros((self.job_num, 9))
        self.lowest_speed = 1


        ## Action Space for the agent
        self.action_space = Discrete(self.buffer_length)

        self.observation_space_dict = Dict({
            'action_mask': Box(0, 1, shape=(self.buffer_length,)),
            'avail_actions': Box(-np.inf, np.inf, shape=(self.buffer_length,)),
            'MCenv': Dict({
                'Online_Buffer': Box(low=-2, high=np.inf ,shape=(self.buffer_length,)),
                'C_jobs': MultiBinary(self.buffer_length),
                'RemLaxity_jobs': Box(low=-np.inf, high=np.inf, shape=(self.buffer_length, 2)),
                'ProcessorSpeed':Box(low=np.array([0.]), high=np.array([np.inf])),
            })
        })

        self.observation_space= flatten_space(self.observation_space_dict)

        self.degradationschedule = []

        self.workbuffer = np.zeros((self.buffer_length, 30))
        for row in self.workbuffer:      ## Making Deadline infinite for dummy jobs
            row[1] = np.inf
        self.online_buffer = [-1] * self.buffer_length
        self.updateBuffer()


        ## Action assignment
        self.action_mask = np.ones(self.buffer_length)

        self.action_assignments = np.ones(self.buffer_length)


        self._update_available()

    def updateBuffer(self):
        self._update_workload()
        self.current_released_jobs = 0
        self.online_buffer = [-1]*self.buffer_length
        self.workload = self.workload[self.workload[:, 5].argsort()]
        counter = 0
        for idx, val in enumerate(self.workload):
            if(val[6] == 1 and val[7] != 1 and val[8] != 1):
                self.online_buffer[counter] = idx
                counter += 1
            if(counter >= self.buffer_length):
                break
        self.current_released_jobs = counter

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def degradationHappens(self):
        if(random.random() < self.degradation_threshold):
            self.speed = np.random.uniform(low=self.lowest_speed, high=1)
        else:
            self.speed = 1

    def step(self, action):
        self.done = self._done()  ## Terminate if Done
        prev_workload = np.copy(self.workload)
        self.reward = 0
        self.info["timebefore"] = self.time
        self.info["workload_before"] = self.workload
        self.info["onlinebuffer_before"] = self.online_buffer
        self.info["done"] = self.done

        if self.current_released_jobs == 0:
            self.time += 1
            self.updateBuffer()
            self._update_available()
            self.state = self._get_obs()
            self.reward = 0
            self.done = self._done()
            return self.state, self.reward, self.done, self.info

        current_action = self.online_buffer[action]
        self.info["currently_released_jobs"] = self.current_released_jobs
        self.info["onlinebuffer_withjobs"] = self.online_buffer
        self.info["ACTION_INDEX"] = current_action
        self.info["current_action"] = self.workload[current_action, :]

        if current_action == -1:
            self.state = self._get_obs()
            self.reward = -25
            self.done = self._done()
            return self.state, self.reward, self.done, self.info
        else:
            degradaded_action = True
            if self.speed == 1:
                degradaded_action = False

            if self.workload[current_action, 7] == 1 or self.workload[current_action, 8] == 1:
                self._update_available()
                self.state = self._get_obs()
                self.reward = -10
                self.done = self._done()
                return self.state, self.reward, self.done, self.info

            self.time = max(self.time, self.workload[current_action, 0])

            low_jobs_count = 0
            hi_jobs_count = 0
            for val in self.online_buffer:
                if val != -1:
                    if self.workload[current_action,3] == 1:
                        hi_jobs_count += 1
                    else:
                        low_jobs_count += 1

            ## Inverse Rank of Laxity in online buffer
            laxity_rank = 0
            for idx,val in enumerate(self.online_buffer):
                if val == current_action and val != 1:
                    laxity_rank = self.buffer_length - idx
                    break
            # laxity_rank = self.buffer_length - np.where(self.online_buffer == current_action)

            deadline_flag = False
            while (self.workload[current_action, 4] > 0):
                if (self.time >= self.workload[current_action, 1]):  ## Deadline is met and job hasn't finished
                    deadline_flag = True
                    break

                self.workload[current_action, 4] -= self.speed
                if(self.speed < 1):
                    self.degradationschedule.append((self.time, self.speed))

                self.time += 1
                self.degradationHappens()

            if(deadline_flag == False):
                self.workload[current_action, 8] = 1 ## Mark job as executed
                #### CALCULATE REWARD ####
                if self.current_released_jobs <= 1:
                    self.reward = 0
                else:
                    if degradaded_action == False:     ## Action was taken under normal speed
                        self.reward += laxity_rank
                    else:                              ## Action was taken under degraded speed
                        if low_jobs_count == 0:
                            self.reward += laxity_rank
                        elif hi_jobs_count == 0:
                            self.reward += laxity_rank
                        elif self.workload[current_action, 3] == 1:
                            self.reward += self.buffer_length + laxity_rank
                        else:
                            self.reward -= 1

                ## Upadting Laxity of jobs
                for row in self.workload:
                    row[5] = row[1] - row[4]
                self.workload[:, 6][self.time >= self.workload[:, 0]] = 1  ##update currently released jobs
                expired_jobs = (self.time >= self.workload[:, 1]) * (1 - self.workload[:, 8]).astype(bool)  ##Already Expired
                self.workload[:, 7][expired_jobs] = 1
                starving_jobs = (self.time + self.workload[:, 2] > self.workload[:, 1]) * (1 - self.workload[:, 8]).astype(bool)  ## Already starved
                self.workload[:, 7][starving_jobs] = 1
                self.done = self._done()
            else:  ## If a job was chosen and it failed to meet deadline
                self.workload[current_action, 7] = 1
                self.reward = -1

        self.updateBuffer()
        self._update_available()
        self.state = self._get_obs()
        self.done = self._done()
        self.info["done"] = self.done
        self.info["reward"] = self.reward
        self.info["onlinebuffer_after"] = self.online_buffer
        self.info["ACTION_INDEX"] = current_action
        self.info["current_action"] = self.workload[current_action, :]
        self.info["workload_after"] = self.workload
        self.info["timeafter"] = self.time
        return self.state, self.reward, self.done, self.info

    def _update_available(self):
        self.degradationHappens()
        for idx, val in enumerate(self.online_buffer):
            if(val == -1):
                self.action_mask[idx] = 0

    def _update_workload(self):
        ## Upadting Laxity of jobs
        for row in self.workload:
            row[5] = row[1] - row[4]
        ## Updating Released jobs to be available and expiring the starved jobs
        self.workload[:, 6][self.time >= self.workload[:, 0]] = 1     ##Check jobs that are released ( t > rj) making their release status = 1
        self.workload[:, 7][self.time + self.workload[:, 2] > self.workload[:, 1]] = 1  #Expired Jobs (their processing time > deadline) , label starvation status = 1

    def _get_obs(self):
        try:
            self.workbuffer = np.zeros((self.buffer_length, self.workload.shape[1]))
            for row in self.workbuffer:      ## Making Deadline infinite for dummy jobs
                row[1] = np.inf
            counter = 0
            for row in self.online_buffer:
                if row != -1:
                    self.workbuffer[counter] = self.workload[row]
                    counter += 1
                if counter >= self.buffer_length:
                    break
        except:
            print("ERROR with work buffer")

        obs_dict = dict({
            'action_mask': self.action_mask,
            'avail_actions': self.action_assignments,
            'MCenv': dict({
                'Online_Buffer': np.array(self.online_buffer),
                'C_jobs': np.array(self.workbuffer[:, 3]).flatten(),            ## Criticality Column
                'RemLaxity_jobs': np.array(self.workbuffer[:, 4:6]),                   ## Remaining time and Adjusted Priority
                'ProcessorSpeed': np.array([self.speed]).flatten()
            })
        })
        return flatten(self.observation_space_dict, obs_dict)

    def reset(self):

        self.time = 0

        self.job_num = 30
        self.total_load = np.random.uniform(low=0.2, high=1)  # env_config['total_load']
        self.lo_per = np.random.uniform(low=0, high=1 - 2 / self.job_num)  # env_config['lo_per']
        self.job_density = np.random.randint(low=self.job_num*(1/4), high=self.job_num*(1/2))  # env_config['job_density']

        ## workload is the creation of the job list from data generation
        ## The 9 columns stand for (ReleaseTime,Deadline,WCET,Criticality,RemainingTime,AdjustedPriority,ReleaseStatus,StarvationStatus,ExecutionStatus)
        # The first 6 are generated from the dataset filtration and the rest are status parameters
        self.seed()
        self.speed = 1
        self.degradation_threshold = np.random.uniform(low=0.05, high=0.95)
        self.filter_threshold = 0.9

        self.workload = np.zeros((self.job_num, 9))
        self.lowest_speed = 1

        self.done = False
        self.info = {}
        self.reward = 0
        self.state = self._get_obs()
        self.degradationschedule = []

        self.workbuffer = np.zeros((self.buffer_length, 30))
        for row in self.workbuffer:      ## Making Deadline infinite for dummy jobs
            row[1] = np.inf
        self.online_buffer = [-1] * self.buffer_length

        self.updateBuffer()

        self.action_mask = np.ones(self.buffer_length)

        self.action_assignments = np.ones(self.buffer_length)

        self._update_available()
        return self.state

    def _done(self):
        arr = []
        for row in self.workload:
            if (row[7] == 1 or row[8] == 1):
                arr.append(0)
        if (len(arr) == len(self.workload)):
            return True
        else:
            return False