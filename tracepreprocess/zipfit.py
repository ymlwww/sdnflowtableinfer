'''
zipfit is to fit the json data to three models and estimate the alpha for the total
distrbution of a single file
'''


import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.optimize import curve_fit
 
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def func3(x, a, b, c, d):
    return a * np.exp(-b * pow(x,c)) + d

def func2(x, a, b, c):
    return a * pow(x, b) + c

alpha  = [1,1.1,1.2,1.3]
trunc = 100
with open("1.json",'r') as load_f:
    #fitting zipf for the trace file
    pkts = json.load(load_f)
    print("total pkt number: "+str(len(pkts)))
    begintime = pkts[0].split(",")[0]
    typenum=0
    frequencydict={} #key is the type num, value is the frequency number
    durationdict={} #key is the type num, value is duration
    for pkt in pkts:
        if int(pkt.split(",")[1])>typenum:
            typenum = int(pkt.split(",")[1])
        if str(pkt.split(",")[1]) in frequencydict:
            frequencydict[str(pkt.split(",")[1])] = frequencydict[str(pkt.split(",")[1])] + 1
        else:
            frequencydict[str(pkt.split(",")[1])] = 1
        if str(pkt.split(",")[1]) in durationdict:
            durationdict[str(pkt.split(",")[1])].append(pkt.split(",")[0])
        else:
            durationdict[str(pkt.split(",")[1])] = [pkt.split(",")[0]]
    durationlist=[]
    for v in durationdict.values():
        if len(v)>1:
            durationlist.append(float(v[len(v)-1])-float(v[0]))            
    frequencylist=[]
    for v in frequencydict.values():
        tmp = v*1.0
        frequencylist.append(tmp)
    frequencylist = sorted(frequencylist,reverse=True)
    freqlist =[c/len(pkts) for c in frequencylist]
    x = range(1,len(freqlist))
    #x = range(1,trunc+1)
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
    x=range(0,len(freqlist))
    for i in range(0,len(alpha)):
        ziplist[i] = ziplist[i][0:trunc]
    plt.plot(x,freqlist,label='data')
    '''
    popt, pcov = curve_fit(func, x, freqlist)
    plt.plot(x, func(x, *popt), label='fit with a * np.exp(-b * x) + c')
    popt2, pcov2 = curve_fit(func2, x, freqlist)
    plt.plot(x, func2(x, *popt2) , label='fit with a * pow(x, b) + c')
    popt3, pcov3 = curve_fit(func3, x, freqlist)
    plt.plot(x, func3(x, *popt3) , label='fit with a * np.exp(-b * pow(x,c)) + d')
    print(popt3)
    '''
    x=range(0,len(freqlist))
    for i in range(0,len(alpha)):
        print(len(x))
        print(len(ziplist[i]))
        plt.plot(x,ziplist[i],label="zipf"+str(alpha[i]))
    

    '''
    plt.hist(durationlist, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.ylabel('number of flows')
    plt.xlabel('duration time/ns')
    plt.title('flow duration frequency histgram')
    plt.show()
    averageduration=0.0
    for duration in durationlist:
        averageduration = averageduration + duration
    averageduration = averageduration/len(durationlist)
    print("average duration time: "+str(averageduration))
    '''
    plt.ylabel('relative frequency')
    plt.xlabel('flow id')
    plt.title('flow size frequency histgram')
    print("total type number: "+ str(typenum))
    plt.legend()
    plt.show()