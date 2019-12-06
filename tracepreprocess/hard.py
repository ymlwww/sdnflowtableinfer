'''
implement the truncation method mentioned in the paper
'''

import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.optimize import curve_fit
import random
import scipy.io as sio
alpha  = [1.2,1.3,1.4,1.5]
trunc = 50
k = 3
with open(str(k)+".json",'r') as load_f:
    #fitting zipf for the trace file
    pkts = json.load(load_f)
    print(len(pkts))
    print(pkts[len(pkts)-1].split(",")[0])
    begintime = pkts[0].split(",")[0]
    typenum=0
    currentlen=0
    frequencydict={} #key is the type num, value is the frequency number
    startpoint = random.randint(1,len(pkts)-10000)
    endpoint = startpoint+10000
    for i in range(startpoint,endpoint):
        pkt = pkts[i]
        currentlen= currentlen + 1
        if str(pkt.split(",")[1]) in frequencydict:
            frequencydict[str(pkt.split(",")[1])] = frequencydict[str(pkt.split(",")[1])] + 1
        else:
            frequencydict[str(pkt.split(",")[1])] = 0        
    frequencylist=[]
    for v in frequencydict.values():
        frequencylist.append(v)
    frequencylist = sorted(frequencylist,reverse=True)
    freqlist =[c/currentlen for c in frequencylist]
    x = range(1,len(freqlist)+1)
    #x = range(1,len(pkts)+1)
    ziplist = []
    for skewness in alpha:
        zipsum=0
        tmplist=[]
        for i in x:
            zipsum = zipsum + pow(i,-skewness)
        for i in x:
            tmplist.append(pow(i,-skewness)/zipsum)
        ziplist.append(tmplist)
    freqlist = freqlist[0:trunc]
    for i in range(0,len(alpha)):
        ziplist[i] = ziplist[i][0:trunc]
    x=range(0,len(freqlist))
    plt.plot(x,freqlist,label='data')
    for i in range(0,len(alpha)):
        plt.plot(x,ziplist[i],label="zipf"+str(alpha[i]))
    plt.ylabel('relative frequency')
    plt.xlabel('flow id')
    plt.title('flow size frequency histgram')
    print(str(startpoint) + " "+ str(endpoint))
    plt.legend()
    plt.show()
    tracetime=[]
    traceid=[]
    for i in range(startpoint,endpoint):
        tracetime.append(float(pkts[i].split(",")[0])*1000)
        traceid.append(int(pkts[i].split(",")[1]))
    normaltracetime=[]
    filestarttime = tracetime[0]
    for item in tracetime:
        normaltracetime.append(item-filestarttime)
    #traceid rename
    tracerenamedict={}
    newtraceid=[]
    firstid = 0
    for id in traceid:
        if id not in tracerenamedict:
            tracerenamedict[id] = firstid
            firstid = firstid + 1 
        newtraceid.append(tracerenamedict[id])
    trace = normaltracetime, newtraceid
    sio.savemat(str(k)+'hard.mat', {'trace':trace})
