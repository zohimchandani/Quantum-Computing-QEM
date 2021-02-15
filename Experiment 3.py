from qiskit import Aer, IBMQ, execute
from qiskit.tools.monitor import job_monitor
from qiskit import *
import random 
from qiskit.providers.aer import noise
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
import qiskit.tools.jupyter  
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
from qiskit import execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info.operators import Kraus, SuperOp
from qiskit.providers.aer import QasmSimulator
from qiskit.tools.visualization import plot_histogram
%qiskit_job_watcher
%load_ext jupyternotify
import operator 
from collections import Counter
from scipy.optimize import curve_fit 
from matplotlib import pyplot as plt 
from random import randrange

simulator = Aer.get_backend('qasm_simulator')

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.providers.ibmq import least_busy
from qiskit import ClassicalRegister, QuantumRegister


import statistics
import dill

import multiprocessing


def noise(q , p): #single qubit noise
   

   r1 =random.random()
   counter1 = 0
   
   if r1 >= 0 and r1 <= p/3 : 
       circ.x(q)
       counter1 = counter1 + 1
   
   elif r1 > p/3 and r1 <= ((2*p)/3):
       circ.y(q)
       counter1 = counter1 + 1
       
   elif r1 > ((2*p)/3) and r1 <= ((3*p)/3):
       
       circ.z(q)
       counter1 = counter1 + 1

   else:
       circ.i(q)
       
   return counter1

def noise2(q1 , q2 ,  p): 
   

   r1 =random.random()
   counter1 = 0
   
   v1 = (p/3)*(1-p)
   v2 = ((p**2)/9)
   
   
   if r1 > 0 and r1 <= v1:
       circ.i(q1)
       circ.x(q2)
       counter1 = counter1 + 1

   
   elif r1> v1 and r1 <= 2*v1:
       circ.i(q1)
       circ.y(q2)
       counter1 = counter1 + 1
       
   elif r1> 2*v1 and r1 <= 3*v1:
       circ.i(q1)
       circ.z(q2)
       counter1 = counter1 + 1  
       
   elif r1> 3*v1 and r1 <= 4*v1:
       circ.x(q1)
       circ.i(q2)
       counter1 = counter1 + 1
       
   elif r1> 4*v1 and r1 <= 5*v1:
       circ.y(q1)
       circ.i(q2)
       counter1 = counter1 + 1
       
   elif r1> 5*v1 and r1 <= 6*v1:
       circ.z(q1)
       circ.i(q2)
       counter1 = counter1 + 1
       
   elif r1> 6*v1 and r1 <= ((6*v1)+v2):
       circ.x(q1)
       circ.x(q2)
       counter1 = counter1 + 2
                                       
   elif r1> ((6*v1)+v2) and r1 <= ((6*v1)+(2*v2)):
       circ.x(q1)
       circ.y(q2)
       counter1 = counter1 + 2
       
   elif r1> ((6*v1)+(2*v2)) and r1 <= ((6*v1)+(3*v2)):
       circ.x(q1)
       circ.z(q2)
       counter1 = counter1 + 2
   
   elif r1> ((6*v1)+(3*v2)) and r1 <= ((6*v1)+(4*v2)):
       circ.y(q1)
       circ.x(q2)
       counter1 = counter1 + 2       
       
   elif r1> ((6*v1)+(4*v2)) and r1 <= ((6*v1)+(5*v2)):
       circ.y(q1)
       circ.y(q2)
       counter1 = counter1 + 2   
       
   elif r1> ((6*v1)+(5*v2)) and r1 <= ((6*v1)+(6*v2)):
       circ.y(q1)
       circ.z(q2)
       counter1 = counter1 + 2    
       
   elif r1> ((6*v1)+(6*v2)) and r1 <= ((6*v1)+(7*v2)):
       circ.z(q1)
       circ.x(q2)
       counter1 = counter1 + 2  
       
   elif r1> ((6*v1)+(7*v2)) and r1 <= ((6*v1)+(8*v2)):
       circ.z(q1)
       circ.y(q2)
       counter1 = counter1 + 2  
       
   elif r1> ((6*v1)+(8*v2)) and r1 <= ((6*v1)+(9*v2)):
       circ.z(q1)
       circ.z(q2)
       counter1 = counter1 + 2   
       
   else: 
       circ.i(q1)
       circ.i(q2)
   
       
   return counter1

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

def define_val(p):
   
   gamma = ((4*p) + 6) / (6 - (8*p))
   p1 = (12 - (4*p) ) / ((8*p) + 12)
   p2 = (4*p) / ((8*p) + 12)
   
   return p1,p2,gamma

p13, p23, gamma = define_val(10**(-3))

%%time
%%notify

# #ibmqx2
single_q_gateerror = 7.64 * 10**(-4)
double_q_gateerror = 1.14 * 10**(-2)

p1, p2, gamma1 = define_val(single_q_gateerror)
p12,p22,gamma2 = define_val(double_q_gateerror)

no_of_qubits = 5
shots = 1000
runs1= 100

circuits = {}


for n in range(1, no_of_qubits+1):
   
   c = {}
   for i in range(2**n):
       string = "{0:0"+str(n)+"b}"
       c[string.format(i)] = 0
   
   for r1 in range(runs1):
       


       circ = QuantumCircuit(n,n)
       circ.h(0)
       noise(0 ,  single_q_gateerror)  


       for idx in range(1,n):
           circ.cx(0,idx)      
           noise2(0 , idx  , double_q_gateerror )


       circ.barrier(range(n))
       circ.measure(range(n), range(n))    
       
       circuits[n,r1] = circ


job_exp = execute(list(circuits.values()) , backend = simulator, shots = shots)
results = job_exp.result()

noisycounts={}
n = 1
r = 0 

for i in range(no_of_qubits*runs1):

   c = {}
   for t in range(2**n):
       string = "{0:0"+str(n)+"b}"
       c[string.format(t)] = 0
       
       
   nonzerocounts = results.get_counts(i)
   
   for key in nonzerocounts:
       c[key] = nonzerocounts[key]
       
   noisycounts[n,r] = c
   
   r += 1
   
   if r== runs1:
       r=0 
       n+=1
       

nop_sim = {}
np_std_sim={}


for n in range(1, no_of_qubits+1):
   
   state = '0'*n
   state1 = '1'*n

   psum ={}

   for r1 in range(runs1):
       
       p0 = (noisycounts[n,r1][state])/shots
       p1 = (noisycounts[n,r1][state1])/shots
       psum[r1] = p0+p1
       

   nop_sim[n]= statistics.mean(psum.values())
   np_std_sim[n] = statistics.stdev(psum.values())

%%time
#%%notify
##final 



runs2 = 1300

mitigatedcounts={}
correction_sum = {}

mitigated_circuits = {}
gamma_sum={}
gamma_sum2={}

for n in range(1, no_of_qubits+1):

   for r2 in range(runs2):

       correction_sum[n,r2]=0
       gamma_sum[n,r2]=0
       gamma_sum2[n,r2]=0
       

       circ = QuantumCircuit(n,n)
       
       circ.h(0)
       
       noise(0 ,  single_q_gateerror)
       
       cs, gs = correction(0,p1,p2)
       correction_sum[n,r2] += cs 
       gamma_sum[n,r2] += gs
       

       for idx in range(1,n):
           circ.cx(0,idx)
           
           noise2(0 , idx  , double_q_gateerror )
           
           cs , gs = correction2(0,idx,p12,p22)
           correction_sum[n,r2] += cs 
           gamma_sum2[n,r2] += gs

       circ.barrier(range(n))
       circ.measure(range(n), range(n))
       
       mitigatedcounts[n,r2] = noisycounts[n,randrange(r1+1)]

       if correction_sum[n,r2]==0:    
           pass

       else:
           mitigated_circuits[n,r2] = circ


whole_list = list(mitigated_circuits.values())
v = 75

circuit_chunks = [whole_list[i:i + v] for i in range(0, len(whole_list), v)]
replace = list(mitigated_circuits.keys())

replace_chunks = [replace[i:i + v] for i in range(0, len(replace), v)]


%%time

for j in range(len(circuit_chunks)):
   
   job_exp = execute(circuit_chunks[j] , backend = simulator, shots = shots)
   results = job_exp.result()

   for i in range(len(circuit_chunks[j])):
       
       n = replace_chunks[j][i][0]
       
       c = {}
       for t in range(2**n):
           string = "{0:0"+str(n)+"b}"
           c[string.format(t)] = 0
           
       nonzerocounts = results.get_counts(i)
       
       
       for key in nonzerocounts:
           c[key] = nonzerocounts[key]

       
       mitigatedcounts[replace_chunks[j][i]] = c

mp_sim = {}
mp_std_sim={}


for n in range(1, no_of_qubits+1):
   
   state = '0'*n
   state1 = '1'*n

   psum={}
   cs={}
   inv1={}
   inv2={}

   for r2 in range(runs2):
       
       p0 = (mitigatedcounts[n,r2][state])/shots
       p1 = (mitigatedcounts[n,r2][state1])/shots
       psum[r2] = (p0+p1)* ((-1)**(correction_sum[n,r2]))
       
       cs[r2] = correction_sum[n,r2]
       inv1[r2] = gamma_sum[n,r2]
       inv2[r2] = gamma_sum2[n,r2]
       
      
   cssum = sum(cs.values()) 

   mp_sim[n]= statistics.mean(psum.values()) * (gamma**cssum)

       
   mp_std_sim[n] = statistics.stdev(psum.values())

x_sim = list(nop_sim.keys())
y_sim = list(nop_sim.values())
x1_sim = list(mp_sim.keys())
y1_sim = list(mp_sim.values())  
ynerr_sim = list(np_std_sim.values())
ymerr_sim = list(mp_std_sim.values())

ymerr_sim_above = [1-x for x in y1_sim]

ymerr_sim_below = ymerr_sim

ynerr_sim_above = [1-x for x in y_sim]

ynerr_sim_below = ynerr_sim

a = 'ibmqx2'

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


nop = {}
np_std={}

no_of_qubits=5
runs1=10
shots=1000


for n in range(1, no_of_qubits+1):
   
   state = '0'*n
   state1 = '1'*n

   psum ={}

   for r1 in range(runs1):
       
       p0 = (noisycounts[n,r1][state])/shots
       p1 = (noisycounts[n,r1][state1])/shots
       psum[r1] = p0+p1
       

   nop[n]= statistics.mean(psum.values())
   np_std[n] = statistics.stdev(psum.values())

mp = {}
mp_std={}

no_of_qubits=5

shots=1000
runs2=1300


for n in range(1, no_of_qubits+1):
   
   state = '0'*n
   state1 = '1'*n

   psum={}
   cs={}
   inv1={}
   inv2={}

   for r2 in range(runs2):
       
       p0 = (mitigatedcounts[n,r2][state])/shots
       p1 = (mitigatedcounts[n,r2][state1])/shots
       psum[r2] = (p0+p1)* ((-1)**(correction_sum[n,r2]))
       
       cs[r2] = correction_sum[n,r2]
       inv1[r2] = gamma_sum[n,r2]
       inv2[r2] = gamma_sum2[n,r2]
      
       
   mp[n]= statistics.mean(psum.values()) * (gamma**sum(cs.values()))
       
   mp_std[n] = statistics.stdev(psum.values())

x = list(nop.keys())
y = list(nop.values())
x1 = list(mp.keys())
y1 = list(mp.values())  
ynerr = list(np_std.values())
ymerr = list(mp_std.values())

mp_err_above = [0.9999-x for x in y1]
mp_err_below = [0,0.2,0.27,0.28,0.29]

import seaborn as sns
from matplotlib.pyplot import figure
figure(num=None, figsize=(9,6), dpi=100, facecolor='w', edgecolor='k')
from pylab import *
fontsize = 15


#plt.plot(x, y, color='red', linestyle='dashed', linewidth = 2, marker='*', markerfacecolor='red', markersize=10, label="Noisy-IBMQ Yorktown") 

#plt.plot(x1, y1, color='green', linestyle='dashed', linewidth = 2, marker='*', markerfacecolor='green', markersize=10, label="PEC-Mitigated-IBMQ Yorktown") 

#plt.plot(x1_sim, y1_sim, color='green', linestyle='dashed', linewidth = 2, marker='.', markerfacecolor='green', markersize=10, label="PEC-Mitigated-Simulated") 

#plt.plot(x_sim, y_sim, color='red', linestyle='dashed', linewidth = 2, marker='.', markerfacecolor='red', markersize=10, label="Noisy-Simulated") 




#plt.errorbar(x, y, color='red', linestyle='dashed', linewidth = 2, yerr=ynerr, elinewidth=1, capsize=5, marker='.', markerfacecolor='red', markersize=10, label="Noisy - IBMQ Yorktown ") 
#plt.errorbar(x1, y1, color='green', linestyle='dashed', linewidth = 2, yerr = (mp_err_below , mp_err_above) , elinewidth=1, capsize=5, marker='.', markerfacecolor='green', markersize=10, label="PEC - Mitigated - IBMQ Yorktown")  

plt.errorbar(x_sim, y_sim, color='red', linestyle='dashed', linewidth = 2, yerr = (ynerr_sim_below , ynerr_sim_above)     , elinewidth=1, capsize=5, marker='*', markerfacecolor='red', markersize=10, label="Noisy-Simulated") 
plt.errorbar(x1_sim, y1_sim, color='green', linestyle='dashed', linewidth = 2, yerr = (ymerr_sim_below , ymerr_sim_above), elinewidth=1, capsize=5, marker='*', markerfacecolor='green', markersize=10, label="PEC-Mitigated-Simulated")  



plt.xlabel('Number of qubits in GHZ state', fontsize=15)

plt.ylabel('Sum of probabilities of $0^{\otimes n}$ and $1^{\otimes n}$ states', fontsize=15)


plt.legend(loc = 'best')

plt.xticks([1,2,3,4,5])
plt.ylim(0.5,1.01)     # set the ylim to bottom, top

plt.savefig('exp4.2.png')
