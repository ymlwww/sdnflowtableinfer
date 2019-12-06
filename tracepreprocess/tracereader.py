'''
the tracereader.py is to read the UNIV1 OR UNIV2 trace file.Since the original
file is large and contains too many uncessary information for us. We just extract
some information of them to json files. Note that, the read file operation is time-consuming,
sometimes u need to truncate the original trace file to 100MB and then merge the json files.
'''

from scapy.all import *
import json
from scapy.utils import rdpcap
import os

'''
    total dict to classify flows
    key is the jointed string of the header field such as tcp mac address
    value is the frequency of the flow
'''
totalpktid = 1
newlist=[]   #output file in trace format with pairs <timestamp, flow id>
assignedid=1  #assign each flow an id according the arrival sequence: eg. the first new one is 0, the next new one is 1
dictz={}
initaltime = 0
for pktid in range(100):
    print "deal with "+ ": univ2_pt"+str(totalpktid)+":"+str(pktid)
    prefix = "pk"+ str(totalpktid)
    lastfix = ""
    if pktid == 0:
        lastfix = ""
    else:
        lastfix = str(pktid)
    if not os.path.exists(prefix + lastfix):
        continue
    pkts = rdpcap(prefix + lastfix)
    if pktid == 0:
        initaltime = pkts[0].time
    for p in pkts:
        item=[]
        #item.append(str(p.time))
        if p.haslayer("Ethernet"):
            src_mac = p["Ethernet"].src
            dst_mac = p["Ethernet"].dst
            item.append(src_mac)
            item.append(dst_mac)
            #print "smac: %s" % src_mac
            #print "dmac: %s" % dst_mac
        if p.haslayer("IP"):
            src_ip = p["IP"].src
            dst_ip = p["IP"].dst
            item.append(src_ip)
            item.append(dst_ip)
            #print "sip: %s" % src_ip
            #print "dip: %s" % dst_ip
        elif p.haslayer("ARP"):
            src_ip = p["ARP"].psrc
            dst_ip = p["ARP"].pdst
            item.append(src_ip)
            item.append(dst_ip)
            #print "ARPsrc: %s" % src_ip
            #print "ARPdst: %s" % dst_ip
        if p.haslayer("TCP"):
            sport = p["TCP"].sport
            dport = p["TCP"].dport
            item.append(str(sport))
            item.append(str(dport))
            #print "sport: %s" % sport
            #print "dport: %s" % dport
        elif p.haslayer("UDP"):
            sport = p["UDP"].sport
            dport = p["UDP"].dport
            item.append(str(sport))
            item.append(str(dport))
            #print "sport: %s" % sport
            #print "dport: %s" % dport
        #print(len(item))    
        tmp = ".".join(item)
        if not dictz.has_key(tmp):
            dictz[tmp] = assignedid
            assignedid = assignedid + 1
        #newlist.append(str(p.time)+","+str(dictz[tmp]))
        newlist.append(str(p.time-initaltime)+","+str(dictz[tmp]))
#print trace list
with open(str(totalpktid)+".json",'a') as outfile:
    json.dump(newlist,outfile,ensure_ascii=False)
    outfile.write('\n')
