from filter import generateData
import numpy as np
import pickle

## File used to generate datasets
np.set_printoptions(suppress=True)
# Default Values
total_load = [0.3,0.6,0.9]
lo_per = [0.2,0.5,0.8]
job_density = [5]
job_num = 30
threshold = 0.8

i = 0
for load in total_load:
    for loper in lo_per:
        for density in job_density:
            print("new dataset")
            metaf = open('dataset'+str(i)+'metadata.pickle','wb')
            pickle.dump((load,loper,density),metaf)
            tf = open('dataset'+str(i)+'.pickle','wb')
            pickle.dump(' ', tf)
            for trial in range(10000):
                workload_instance, minimumspeed = generateData(load,loper,density,job_num,threshold)

                f = open('dataset'+str(i)+'.pickle','ab')
                pickle.dump((workload_instance,minimumspeed,len(workload_instance)), f)
            print("DONE ONE DATASET")
            i += 1


# for i in range(8):
#     print(i)
#     f = open('dataset'+str(i)+'.pickle','rb')
#     metaf = open('dataset'+str(i)+'metadata.pickle','rb')
#
#     while 1:
#         try:
#             print(pickle.load(metaf))
#         except EOFError:
#             break
#
#
#     while 1:
#         try:
#             print(pickle.load(f))
#             print('---')
#         except EOFError:
#             break