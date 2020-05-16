from qiskit import Aer, IBMQ, execute
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor

from qiskit import *
import random 

from qiskit.providers.aer import noise
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor
import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
%matplotlib inline
import math
from qiskit import Aer
from qiskit.visualization import plot_state_city
from qiskit.visualization import plot_histogram
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import iplot_histogram
import qiskit.tools.jupyter  
import matplotlib.pyplot as plt
%matplotlib inline
from qiskit.visualization import plot_circuit_layout
import numpy as np
from qiskit.visualization import plot_histogram, plot_gate_map, plot_circuit_layout
from qiskit.tools.monitor import job_monitor
from math import pi
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer
from qiskit.providers.ibmq import least_busy
import pandas as pd
import numpy as np
from qiskit import execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info.operators import Kraus, SuperOp
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.visualization import plot_histogram

import jupyternotify

from qiskit import execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Kraus, SuperOp
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.visualization import plot_histogram

from random import randrange
import statistics


%qiskit_job_watcher

def fidelity(counts): #only works for GHZ states!!
    
    
    value=1/math.sqrt(2)
    
    terms = list(counts.keys())
    terms.sort()
    prob = [counts[terms[i]] for i in range(len(terms))]
    state_vector=  np.array(prob)/shots 
    state_vector = np.sqrt(state_vector)

    desired_vector= state_vector*0
    desired_vector[0]= value
    desired_vector[len(terms)-1]= value
    
    f=state_fidelity(desired_vector,state_vector) 
    
    return f


import dill 
import time 
%load_ext jupyternotify


def correction(q, p1, p2):
    
    r2 =random.random()
    counter2 = 0 
    gf=0
    
    if r2>0 and r2<= p2: 
        circ.x(q)
        counter2 = counter2 + 1
        gf = gf + 1

    elif r2>p2  and r2 <= (2*p2):
        circ.y(q)
        counter2 = counter2 + 1
        gf = gf + 1

    elif r2> (2*p2) and  r2 <= (3*p2):
        circ.z(q)
        counter2 = counter2 + 1
        gf = gf + 1
        
    else:
        circ.i(q)


    return counter2,gf


def correction2(q1, q2,  p1, p2):
    
    h1 = p1*p2
    h2 = p2**2
    
    r2 =random.random()
    counter2 = 0 
    gf=0
    
    
    
    if r2 > 0 and r2 <= h1:
        circ.i(q1)
        circ.x(q2)
        counter2 = counter2 + 1
        gf = gf + 1


    elif r2 >h1  and r2 <= (2*h1):
        circ.i(q1)
        circ.y(q2)
        counter2 = counter2 + 1
        gf = gf + 1

    elif r2> (2*h1)  and  r2 <= (3*h1):
        circ.i(q1)
        circ.z(q2)
        counter2 = counter2 + 1
        gf = gf + 1

    elif r2> (3*h1)  and  r2 <= (4*h1):
        circ.x(q1)
        circ.i(q2)
        counter2 = counter2 +1
        gf = gf + 1
        
    elif r2> (4*h1)  and  r2 <= (5*h1):
        circ.y(q1)
        circ.i(q2)
        counter2 = counter2 + 1
        gf = gf + 1
        
    elif r2> (5*h1)  and  r2 <= (6*h1):
        circ.z(q1)
        circ.i(q2)
        counter2 = counter2 + 1
        gf = gf + 1
        
    elif r2> (6*h1)  and  r2 <= ((6*h1)+h2):
        circ.x(q1)
        circ.x(q2)
        counter2 = counter2 + 2
        gf = gf + 1
        
    
    elif r2> ((6*h1)+h2)  and  r2 <= ((6*h1)+(2*h2)):
        circ.x(q1)
        circ.y(q2)
        counter2 = counter2 + 2
        gf = gf + 1
        
    elif r2> ((6*h1)+(2*h2))  and  r2 <= ((6*h1)+(3*h2)):
        circ.x(q1)
        circ.z(q2)
        counter2 = counter2 + 2
        gf = gf + 1
        
    elif r2> ((6*h1)+(3*h2))  and  r2 <= ((6*h1)+(4*h2)):
        circ.y(q1)
        circ.x(q2)
        counter2 = counter2 + 2
        gf = gf + 1
        
    elif r2> ((6*h1)+(4*h2))  and  r2 <= ((6*h1)+(5*h2)):
        circ.y(q1)             
        circ.y(q2)
        counter2 = counter2 + 2
        gf = gf + 1
    
    elif r2> ((6*h1)+(5*h2))  and  r2 <= ((6*h1)+(6*h2)):
        circ.y(q1)
        circ.z(q2)
        counter2 = counter2 + 2
        gf = gf + 1
        
    elif r2> ((6*h1)+(6*h2))  and  r2 <= ((6*h1)+(7*h2)):
        circ.z(q1)
        circ.x(q2)
        counter2 = counter2 + 2
        gf = gf + 1
        
    elif r2> ((6*h1)+(7*h2))  and  r2 <= ((6*h1)+(8*h2)):
        circ.z(q1)
        circ.y(q2)
        counter2 = counter2 + 2
        gf = gf + 1
        
    elif r2> ((6*h1)+(8*h2))  and  r2 <= ((6*h1)+(9*h2)):
        circ.z(q1)
        circ.z(q2)
        counter2 = counter2 + 2
        gf = gf + 1
        
    else:
        circ.i(q1)
        circ.i(q2)
        
    return counter2,gf

provider = IBMQ.load_account()
provider.backends()
simulator = Aer.get_backend('qasm_simulator')

def values(p):
    
    gamma = ((4*p) + 6) / (6 - (8*p))
    p1 = (12 - (4*p) ) / ((8*p) + 12)
    p2 = (4*p) / ((8*p) + 12)
    
    return p1,p2,gamma

b1 = provider.get_backend('ibmq_16_melbourne')
b2 = provider.get_backend('ibmq_rome')
b3 = provider.get_backend('ibmq_london')
b4 = provider.get_backend('ibmq_burlington')
b5 = provider.get_backend('ibmq_essex')
b6 = provider.get_backend('ibmq_ourense')
b7 = provider.get_backend('ibmq_vigo')
b8 = provider.get_backend('ibmqx2')


backend  = b8
a = 'ibmqx2'


# #melbourne
# single_q_gateerror = 1.31 * 10**(-3)
# double_q_gateerror = 2.55 * 10**(-2)

#rome
# single_q_gateerror = 3.66 * 10**(-4)
# double_q_gateerror = 7.68 * 10**(-3)


#london
# single_q_gateerror = 6.44 * 10**(-3)
# double_q_gateerror = 6.15 * 10**(-2)

# burlington
# single_q_gateerror = 4.98 * 10**(-4)
# double_q_gateerror = 1.40 * 10**(-2)


#essex
# single_q_gateerror = 5.30 * 10**(-4)
# double_q_gateerror = 1.45 * 10**(-2)


# #oursene
# single_q_gateerror = 3.35 * 10**(-4)
# double_q_gateerror = 8.73 * 10**(-3)

# #vigo
# single_q_gateerror = 4.43 * 10**(-4)
# double_q_gateerror =  8.81 * 10**(-3)

# #ibmqx2
single_q_gateerror = 7.64 * 10**(-4)
double_q_gateerror = 1.14 * 10**(-2)



p1,p2,gamma1= values(single_q_gateerror)
p12,p22,gamma2 = values(double_q_gateerror)

p13, p23, gamma = values(10**(-3))

# %%time
# %%notify
 
# no_of_qubits = 5
# shots = 1000
# runs1= 10

# noisycounts={}

# circuits = {}

# for n in range(1, no_of_qubits+1):
    
#     for r1 in range(runs1):

#         circ = QuantumCircuit(n,n)
#         circ.h(0)

#         for idx in range(1,n):
#             circ.cx(0,idx)      

#         circ.barrier(range(n))
#         circ.measure(range(n), range(n))
        
#         circuits[n,r1] = circ


# job_exp = execute(list(circuits.values()) , backend = backend, shots = shots)
# #job_exp = execute(list(circuits.values()) , backend = simulator, shots = shots)
# results = job_exp.result()

# noisycounts={}
# n = 1
# r = 0 

# for i in range(no_of_qubits*runs1):
    

    
#     c = {}
#     for t in range(2**n):
#         string = "{0:0"+str(n)+"b}"
#         c[string.format(t)] = 0
        
        
#     nonzerocounts = results.get_counts(i)
    
#     for key in nonzerocounts:
#         c[key] = nonzerocounts[key]
        
#     noisycounts[n,r] = c
    
#     r += 1
    
#     if r== runs1:
#         r=0 
#         n+=1
        

# %%time
# #%%notify
# ##final 


# runs2= 1300

# mitigatedcounts={}
# correction_sum = {}
# gamma_sum = {}
# gamma_sum2 = {}
# mitigated_circuits = {}


# for n in range(1, no_of_qubits+1):

#     for r2 in range(runs2):

#         correction_sum[n,r2]=0
#         gamma_sum[n,r2]=0
#         gamma_sum2[n,r2]=0

        

#         circ = QuantumCircuit(n,n)
        
#         circ.h(0)
#         cs, gs = correction(0,p1,p2)
#         correction_sum[n,r2] += cs 
#         gamma_sum[n,r2] += gs
        

#         for idx in range(1,n):
#             circ.cx(0,idx)
            
#             cs , gs = correction2(0,idx,p12,p22)
#             correction_sum[n,r2] += cs 
#             gamma_sum2[n,r2] += gs

#         circ.barrier(range(n))
#         circ.measure(range(n), range(n))
        
#         mitigatedcounts[n,r2] = noisycounts[n,randrange(r1+1)]

#         if correction_sum[n,r2]==0:    
#             pass

#         else:
#             mitigated_circuits[n,r2] = circ


# whole_list = list(mitigated_circuits.values())
# v = 75

# circuit_chunks = [whole_list[i:i + v] for i in range(0, len(whole_list), v)]
# replace = list(mitigated_circuits.keys())

# replace_chunks = [replace[i:i + v] for i in range(0, len(replace), v)]


# for j in range(len(circuit_chunks)):
    
#     job_exp = execute(circuit_chunks[j] , backend = backend, shots = shots)
#     #job_exp = execute(circuit_chunks[j] , backend = simulator, shots = shots)
#     results = job_exp.result()

#     for i in range(len(circuit_chunks[j])):
        
#         n = replace_chunks[j][i][0]
        
#         c = {}
#         for t in range(2**n):
#             string = "{0:0"+str(n)+"b}"
#             c[string.format(t)] = 0
            
#         nonzerocounts = results.get_counts(i)
        
        
#         for key in nonzerocounts:
#             c[key] = nonzerocounts[key]

        
#         mitigatedcounts[replace_chunks[j][i]] = c

# import pickle

# with open("noisycounts_" +str(a)+"1.p", 'wb') as fp:
#     pickle.dump(noisycounts, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open("mitigatedcounts_" +str(a)+"1.p", 'wb') as fp:
#     pickle.dump(mitigatedcounts, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open("correction_sum_" +str(a)+"1.p", 'wb') as fp:
#     pickle.dump(correction_sum, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open("gamma_sum_" +str(a)+"1.p", 'wb') as fp:
#     pickle.dump(gamma_sum, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open("gamma_sum2_" +str(a)+"1.p", 'wb') as fp:
#     pickle.dump(gamma_sum2, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open("mitigated_circuits_" +str(a)+"1.p", 'wb') as fp:
#     pickle.dump(mitigated_circuits, fp, protocol=pickle.HIGHEST_PROTOCOL)


import pickle

with open("noisycounts_" +str(a)+"1.p", 'rb') as fp:
    noisycounts = pickle.load(fp) 
    
with open("mitigatedcounts_" +str(a)+"1.p", 'rb') as fp:
    mitigatedcounts = pickle.load(fp) 
    
with open("correction_sum_" +str(a)+"1.p", 'rb') as fp:
    correction_sum = pickle.load(fp) 
    
with open("gamma_sum_" +str(a)+"1.p", 'rb') as fp:
    gamma_sum = pickle.load(fp) 
    
with open("gamma_sum2_" +str(a)+"1.p", 'rb') as fp:
    gamma_sum2 = pickle.load(fp) 
    
with open("mitigated_circuits_" +str(a)+"1.p", 'rb') as fp:
    mitigated_circuits = pickle.load(fp) 


# nop = {}
# np_std={}

# no_of_qubits=5
# runs1=10
# shots=1000


# for n in range(1, no_of_qubits+1):
    
#     state = '0'*n
#     state1 = '1'*n

#     psum ={}

#     for r1 in range(runs1):
        
#         p0 = (noisycounts[n,r1][state])/shots
#         p1 = (noisycounts[n,r1][state1])/shots
#         psum[r1] = p0+p1
        

#     nop[n]= statistics.mean(psum.values())
#     np_std[n] = statistics.stdev(psum.values())



# mp = {}
# mpstore={}
# mp_std={}

# no_of_qubits=5

# shots=1000
# runs2=1300


# for n in range(1, no_of_qubits+1):
    
#     state = '0'*n
#     state1 = '1'*n

#     psum={}
#     cs={}
#     inv1={}
#     inv2={}

#     for r2 in range(runs2):
        
#         p0 = (mitigatedcounts[n,r2][state])/shots
#         p1 = (mitigatedcounts[n,r2][state1])/shots
#         psum[r2] = (p0+p1)* ((-1)**(correction_sum[n,r2]))
        
#         cs[r2] = correction_sum[n,r2]
#         inv1[r2] = gamma_sum[n,r2]
#         inv2[r2] = gamma_sum2[n,r2]
       
#         mpstore[n,r2]=psum.values()
        
#     mp[n]= statistics.mean(psum.values()) * (gamma**sum(cs.values()))
#     mp[n]= statistics.mean(psum.values()) * (gamma1**sum(inv1.values())) * (gamma2**sum(inv2.values()))

        
#     mp_std[n] = statistics.stdev(psum.values())

# import seaborn as sns
# from matplotlib.pyplot import figure
# figure(num=None, figsize=(9,6), dpi=100, facecolor='w', edgecolor='k')
# from pylab import *
# fontsize = 15

# x = list(nop.keys())
# y = list(nop.values())
# x1 = list(mp.keys())
# y1 = list(mp.values())  
# ynerr = list(np_std.values())
# ymerr = list(mp_std.values())

# # plt.plot(x1, y1, color='green', linestyle='dashed', linewidth = 2, marker='.', markerfacecolor='green', markersize=10, label="PEC-Mitigated") 
# # plt.plot(x, y, color='red', linestyle='dashed', linewidth = 2, marker='.', markerfacecolor='red', markersize=10, label="Noisy") 

# plt.errorbar(x, y, color='red', linestyle='dashed', linewidth = 2, yerr=ynerr, elinewidth=1, capsize=5, marker='.', markerfacecolor='red', markersize=10, label="Noisy") 
# #plt.errorbar(x1, y1, color='green', linestyle='dashed', linewidth = 2, yerr=ymerr, elinewidth=1, capsize=5, marker='.', markerfacecolor='green', markersize=10, label="PEC-Mitigated") plt.errorbar(x1, y1, color='green', linestyle='dashed', linewidth = 2, yerr=ymerr, elinewidth=1, capsize=5, marker='.', markerfacecolor='green', markersize=10, label="PEC-Mitigated") 

# plt.errorbar(x1, y1, color='green', linestyle='dashed', linewidth = 2, yerr = (mp_err_below , mp_err_above) , elinewidth=1, capsize=5, marker='.', markerfacecolor='green', markersize=10, label="PEC-Mitigated")  

# ax = gca()
# for tick in ax.xaxis.get_major_ticks():
#     tick.label1.set_fontsize(fontsize)
# for tick in ax.yaxis.get_major_ticks():
#     tick.label1.set_fontsize(fontsize)
    
# plt.xlabel('Number of qubits in GHZ state', fontsize=15)
# plt.ylabel('Measured value of $\mathcal{P}$', fontsize=15)
# plt.legend(loc = 'best')

# # plt.xlim(xmin=0)  
# # plt.ylim(ymin=0)  
# plt.xticks([1,2,3,4,5])

# #plt.savefig('exp4.png')

sum_nf = sum(nop.values())
sum_mf = sum(mp.values())

e_sum_nf = list(np_std.values())
e_sum_nf = [i**2 for i in e_sum_nf]
e_sum_nf = np.sqrt(sum(e_sum_nf))

e_sum_mf = list(mp_std.values())
e_sum_mf = [i**2 for i in e_sum_mf]
e_sum_mf = np.sqrt(sum(e_sum_mf))

print('sum of noisy prob and its error = ' ,sum_nf , e_sum_nf)
print('sum of mitigated prob and its error = ', sum_mf , e_sum_mf)



