import numpy as np
import pickle

## Simply unpickle all data generated on the fly during training
## Find average values of the randomized parameters
np.set_printoptions(suppress=True)

f = open('traindata.pickle','rb')

sizecount = 0
totalcount = 0
min_speed_sum = 0
degradationthreshold_sum = 0
totalload_sum = 0
loper_sum = 0
jobdensity_sum = 0
while 1:
    try:
        row = pickle.load(f)
        totalcount += 1
        sizecount += 1
        min_speed_sum += row[1]
        degradationthreshold_sum += row[2]
        totalload_sum += row[3]
        loper_sum += row[4]
        jobdensity_sum += row[5]
    except EOFError:
        break
    except Exception as e:
        totalcount += 1
        print("Row : ", row)
        continue

print("Size of Training Data : ", sizecount)
print("Average Minimum Speed : ", min_speed_sum/sizecount)
print("Average Degradation Threshold : ", degradationthreshold_sum/sizecount)
print("Average Total Load : ", totalload_sum/sizecount)
print("Average Low Per : ", loper_sum/sizecount)
print("Average Job Density : ", jobdensity_sum/sizecount)
