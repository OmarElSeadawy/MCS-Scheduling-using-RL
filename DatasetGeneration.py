import sys

from filter import generateData
import numpy as np
import pickle

## File used to generate datasets
np.set_printoptions(suppress=True)
# Default Values
total_load = [0.3,0.6,0.9]
lo_per = [0.3,0.5,0.8]
job_density = [5]
job_num = 30
threshold = 0.95

dirname = "newpervsnon/dataset"

i = 0
for load in total_load:
    for loper in lo_per:
        for density in job_density:
            print("new dataset")
            with open(dirname+str(i)+'metadata.pickle','wb') as metaf:
                pickle.dump((load,loper,density),metaf)
            for trial in range(2000):
                if(trial%100==0):
                    print(trial)
                workload_instance, minimumspeed = generateData(load,loper,density,job_num,threshold)
                with open(dirname+str(i)+'.pickle','ab') as f:
                    pickle.dump((workload_instance, minimumspeed, len(workload_instance)), f)
            print("DONE ONE DATASET")
            i += 1


for i in range(1):
    print(i)
    with open(dirname+str(i)+'metadata.pickle','rb') as metaf:
        while 1:
            try:
                print(pickle.load(metaf))
            except EOFError:
                break

    with open(dirname+str(i)+'.pickle','rb') as f:
        while 1:
            try:
                print(pickle.load(f))
                print('---')
            except EOFError:
                break