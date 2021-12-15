#!/usr/bin/env python
# coding: utf-8

#Parameters: # of jobs, total load (<=1), percentage of LO jobs(<=1), Average density (active job number at any instant)
#Output is a set of jobs J of size n by 4 matrix, rj-dj-pj-cj

import numpy as np
import math
import random

## Inspired by Baruah and Guo 2013 paper.
def create_workload(job_num, total_load, lo_per, job_density):

    #### Release Time calculation part

    UB=3.00000000 #Upper_bound for exponential distrbution
    mean=1
    exp_dist=np.random.exponential(mean,(1,job_num-1))#array of Jobnum-1 size each entry is a exponeneital distribution of the entry itself
    exp_dist=np.array(exp_dist)
    exp_dist=exp_dist[0]
    exp_dist[exp_dist>UB]=UB #setting the Upper Bound of the exponential distribution

    release_array =[]
    for i in range (job_num):
        if(i==0):
            release_array.append(0)
        else:
            curr_val=exp_dist[i-1]+release_array[i-1]
            curr_val=round(curr_val,4) #rounding the release value to 4 digits
            release_array.append(curr_val)
    release_array=np.array(release_array)


    #### Criticality level calculation part

    size=(1,job_num)
    criticality_array=np.random.uniform(0,1,size)
    criticality_array[criticality_array<lo_per]=0
    criticality_array[criticality_array>=lo_per]=1
    criticality_array=np.array(criticality_array)
    criticality_array=criticality_array[0]


    #### Deadlines Calculaition part


    def f(x):
        return (math.exp(x) -1 -x*x)
    def f1(x):
        return (math.exp(x)- x)
    def equation_solver(a):
        x0=a
        f_out=f(x0)
        f1_out=f1(x0)
        x1=x0-f_out/f1_out
        x1=round(x1,4)
        return x1
    b= equation_solver(job_density)
    relative_deadline=[]
    deadline_array=[]
    for i in range(job_num):
        unidis=np.random.uniform(0,b)
        relative_deadline.append(math.exp(unidis))

    relative_deadline=np.array(relative_deadline)
    # relative_deadline[1]
    for el in range(job_num):
        deadline_array.append(release_array[el]+relative_deadline[el])
    deadline_array=np.array(deadline_array)


    # ### Processing Time WCET

    Sum_C=max(deadline_array)*total_load
    b1=max(0,Sum_C-sum(relative_deadline)+relative_deadline[0])
    b2=min(relative_deadline[0],Sum_C)
    the_mean=relative_deadline[0]*total_load*max(deadline_array)/sum(relative_deadline)
    beta = 2*(b2-b1)/(the_mean-b1)-2
    excution_Lo=[1]*job_num #creating an empty array
    excution_Lo[0] = random.betavariate(2,beta)*(b2-b1)+b1 #using Beta random distribution
    for i in range(job_num):
        if(i!=0):
            sum1 = 0
            sum2 = 0
            for  j in range(i-1):
                sum1 = sum1+excution_Lo[j]
            k=i+1
            for  k in range(job_num):
                sum1 = sum1+relative_deadline[k]

            for  m in range(i-1):
                sum2 = sum2+excution_Lo[m]

            b1=max(0,Sum_C-sum1) #lowerbound
            b2=min(relative_deadline[i],Sum_C-sum2) #upperbound
            if (b1>b2):
                excution_Lo[i] = 0
                break
            #     %T_excution_Lo(i)=unifrnd(b1,b2);
            the_mean=relative_deadline[i]*total_load*max(deadline_array)/sum(relative_deadline)
            if (the_mean<b1) or (the_mean>b2):
                excution_Lo[i]= 0
                break
            #    %T_excution_Lo(i)=unifrnd(themean-min(b2-themean,themean-b1),themean+min(b2-themean,themean-b1));
            beta = 2*(b2-b1)/(the_mean-b1)-2
            excution_Lo[i] = random.betavariate(2,beta)*(b2-b1)+b1

    #excution_Lo[job_num-1] = min(Sum_C - sum(excution_Lo),relative_deadline[job_num-1])
    processing_array=np.array(excution_Lo)
    for _ in range(job_num):
        deadline_array[_]=round(deadline_array[_],4)
        processing_array[_]=round(processing_array[_],4)



    release_array=release_array.reshape(job_num,1)
    deadline_array=deadline_array.reshape(job_num,1)
    processing_array=processing_array.reshape(job_num,1)
    criticality_array=criticality_array.reshape(job_num,1)
    final=np.concatenate((release_array, deadline_array, processing_array, criticality_array),axis=1)
    final=final[np.argsort(final[:, 1])] #Sorting by Deadline
    #print("#------Shape of final matrix: ", final.shape)
    return final
