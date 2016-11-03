import pandas as pd
from fonction_py.preprocess import *
import numpy as np
import matplotlib.pyplot as plt

import time


start_time = time.time()
print("go")
fields = ['DATE', 'DAY_OFF', 'DAY_DS', 'WEEK_END', 'DAY_WE_DS','TPER_TEAM', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS' ]
data=pd.read_csv("data/train_2011_2012_2013.csv", sep=";", usecols=fields, nrows=10) # LECTURE

x = data[fields[0:-2]] # Data sans les received calls
y = data[fields[-1]] # label = received calls
ass = data[fields[-2]] # ass assignment = differentes categories a pred


x = preprocess(x)


x = pd.concat([x,x['DATE']], axis=1) # test
print(x)

print("--- %s seconds ---" % (time.time() - start_time))



