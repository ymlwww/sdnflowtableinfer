'''
transfer the json file to mat file which can be used in experiments
'''

import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import scipy.io as sio
trunc = 50
with open("2.json",'r') as load_f:
    #fitting zipf for the trace file
    pkts = json.load(load_f)
    print(len(pkts))
    print(pkts[len(pkts)-1].split(",")[0])
    tracetime=[]
    traceid=[]
    for i in range(len(pkts)):
        tracetime.append(float(pkts[i].split(",")[0])*1000)
        traceid.append(int(pkts[i].split(",")[1]))
    tracerenamedict={}
    newtraceid=[]
    firstid = 0
    for id in traceid:
        if id not in tracerenamedict:
            tracerenamedict[id] = firstid
            firstid = firstid + 1 
        newtraceid.append(tracerenamedict[id])
    trace = tracetime, newtraceid
    sio.savemat('univ2trace2.mat', {'trace':trace})