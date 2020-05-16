
circ_depth = 40
circuits = 100

from qiskit import Aer, IBMQ, execute
from qiskit.tools.monitor import job_monitor
from qiskit import *
import random 
from qiskit.providers.aer import noise
import numpy as np
import matplotlib.pyplot as plt
from qiskit import *

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
import operator 
from collections import Counter
from scipy.optimize import curve_fit 
from matplotlib import pyplot as plt 
from random import randrange

simulator = Aer.get_backend('qasm_simulator')

import numpy as np
import matplotlib.pyplot as plt

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
        circ.iden(q)
        
    return counter1

def noise2(q1 , q2 ,  p): 
    

    r1 =random.random()
    counter1 = 0
    
    v1 = (p/3)*(1-p)
    v2 = ((p**2)/9)
    
    
    if r1 > 0 and r1 <= v1:
        circ.iden(q1)
        circ.x(q2)
        counter1 = counter1 + 1

    
    elif r1> v1 and r1 <= 2*v1:
        circ.iden(q1)
        circ.y(q2)
        counter1 = counter1 + 1
        
    elif r1> 2*v1 and r1 <= 3*v1:
        circ.iden(q1)
        circ.z(q2)
        counter1 = counter1 + 1  
        
    elif r1> 3*v1 and r1 <= 4*v1:
        circ.x(q1)
        circ.iden(q2)
        counter1 = counter1 + 1
        
    elif r1> 4*v1 and r1 <= 5*v1:
        circ.y(q1)
        circ.iden(q2)
        counter1 = counter1 + 1
        
    elif r1> 5*v1 and r1 <= 6*v1:
        circ.z(q1)
        circ.iden(q2)
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
        circ.iden(q1)
        circ.iden(q2)
    
        
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
        circ.iden(q)


    return counter2,gf


def correction2(q1, q2,  p1, p2):
    
    h1 = p1*p2
    h2 = p2**2
    
    r2 =random.random()
    counter2 = 0 
    gf=0
    
    
    
    if r2 > 0 and r2 <= h1:
        circ.iden(q1)
        circ.x(q2)
        counter2 = counter2 + 1
        gf = gf + 1


    elif r2 >h1  and r2 <= (2*h1):
        circ.iden(q1)
        circ.y(q2)
        counter2 = counter2 + 1
        gf = gf + 1

    elif r2> (2*h1)  and  r2 <= (3*h1):
        circ.iden(q1)
        circ.z(q2)
        counter2 = counter2 + 1
        gf = gf + 1

    elif r2> (3*h1)  and  r2 <= (4*h1):
        circ.x(q1)
        circ.iden(q2)
        counter2 = counter2 +1
        gf = gf + 1
        
    elif r2> (4*h1)  and  r2 <= (5*h1):
        circ.y(q1)
        circ.iden(q2)
        counter2 = counter2 + 1
        gf = gf + 1
        
    elif r2> (5*h1)  and  r2 <= (6*h1):
        circ.z(q1)
        circ.iden(q2)
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
        circ.iden(q1)
        circ.iden(q2)
        
    return counter2,gf

def define_val(p):
    
    gamma = ((4*p) + 6) / (6 - (8*p))
    p1 = (12 - (4*p) ) / ((8*p) + 12)
    p2 = (4*p) / ((8*p) + 12)
    
    return p1,p2,gamma




n = 5
circ_depth = circ_depth -1
shots = 100
runs = 1000
values = 4 #no of points you want for linear extrapolation 



single_q_gateerror = 10**(-4)
double_q_gateerror = 10**(-2)

p1, p2, gamma1 = define_val(single_q_gateerror)
p12,p22,gamma2 = define_val(double_q_gateerror)


correction_sum={}
counts={}
counts1 = {}
gamma_sum={}
gamma_sum2={}


c = {}
for i in range(2**n):
    string = "{0:0"+str(n)+"b}"
    c[string.format(i)] = 0
    


unitary = [lambda args: circ.iden(*args), lambda args: circ.x(*args), lambda args: circ.y(*args), lambda args: circ.z(*args)
          , lambda args: circ.h(*args), lambda args: circ.s(*args), lambda args: circ.sdg(*args), lambda args: circ.t(*args)
          , lambda args: circ.tdg(*args),lambda args : circ.cx(*args)] 

for circs in range(circuits): 


    gates = []
    index_values = []
    N = len(unitary)
    matrix = np.zeros((n, circ_depth) , dtype = np.int8)
    for x in range(n):
        for y in range(circ_depth):
            index = random.choice(range(N-1))
            matrix[x,y] = index
    for i in range(random.choice(range(10))): #try to put cnot 10 times 
        x1 = random.choice(range(n))
        x2 = x1
        while x1 == x2 : 
            x2 = random.choice(range(n))
        y =  random.choice(range(circ_depth))
        if matrix[x1,y] not in {9, 10 , -1} and matrix[x2,y] not in {9, 10 , -1}:
            matrix[x1,y] = N
            matrix[x2,y] = N-1
            for x in range(min(x1,x2)+1, max(x1,x2)):
                matrix[x,y] = -1
    matrix_done = np.zeros((n, circ_depth) , dtype = np.int8)
    for y in range(circ_depth):
        for x in range(n):
            if matrix_done[x,y] == 0 :
                if matrix[x,y] < N-1 and matrix[x,y] >= 0:
                    gates.append(( unitary[matrix[x,y]] ,  (x, )))
                    index_values.append(matrix[x,y])
                    matrix_done[x,y] = 1 
                else:
                    q1 = x 
                    q2 = x 
                    if matrix[x,y] == 9: 
                        while matrix[q2, y] != 10: 
                            q2 += 1
                    else: 
                        while matrix[q1 ,y] != 9: 
                            q1 += 1

                    for q in range(x , max(q1,q2)+1):
                        matrix_done[q,y] = 1
                    gates.append(( unitary[N-1] ,  (q1 , q2 , )))
                    index_values.append(N-1)



    for r in range(runs):
        noisy_sum = 0 
        correction_sum[circs,r] = 0
        gamma_sum[circs,r] = 0
        gamma_sum2[circs,r] = 0


        seed = random.randint(0,1e10)
        for status in ["ideal", "noisy" , "mitigated"]:
            random.seed(seed)

            circ= QuantumCircuit(n,n)

            for i in range(len(gates)):
                gates[i][0](gates[i][1])

                if status=="noisy" or status=="mitigated":

                    if index_values[i] == (len(unitary) - 1):
                        noisy_sum += noise2(gates[i][1][-2] ,gates[i][1][-1]  , double_q_gateerror )
                    else:
                        noisy_sum += noise(gates[i][1][-1] ,  single_q_gateerror)  
                else:
                    random.random()


                if status=="mitigated":

                    if index_values[i] == (len(unitary) - 1):
                        
                        cs, gs = correction2(gates[i][1][-2] ,gates[i][1][-1]  , p12, p22 )
                        correction_sum[circs,r] += cs
                        gamma_sum2[circs,r] += gs

                        
                    else:
                        cs, gs =correction(gates[i][1][-1] ,  p1 , p2)  
                        
                        correction_sum[circs,r] += cs
                        gamma_sum[circs,r] += gs
                        
                        
                else:
                    random.random()


            circ.barrier(range(n))
            circ.measure(range(n), range(n))


            if status=="ideal":

                c = c.fromkeys(c, 0)

                result = execute(circ, simulator, shots = shots).result()
                nonzerocounts = result.get_counts(circ)
                for key in nonzerocounts:
                    c[key] = nonzerocounts[key]

                counts[circs,r,status] = c


            elif status=="noisy":

                if noisy_sum==0:
                    counts[circs,r,status] = counts[circs,r,"ideal"]

                else:
                    c = c.fromkeys(c, 0)

                    result = execute(circ, simulator, shots = shots ).result()
                    nonzerocounts = result.get_counts(circ)
                    for key in nonzerocounts:
                        c[key] = nonzerocounts[key]

                    counts[circs,r,status] = c


            elif status=="mitigated":

                if correction_sum[circs,r]==0:
                    counts[circs,r,status] = counts[circs,r,"noisy"]

                else:
                    c = c.fromkeys(c, 0)

                    result = execute(circ, simulator, shots = shots ).result()
                    nonzerocounts = result.get_counts(circ)
                    for key in nonzerocounts:
                        c[key] = nonzerocounts[key]

                    counts[circs,r,status] = c




    for m in range(2 , values+1): 

        for r in range (runs):

            circ= QuantumCircuit(n,n)

            noisy_sum = 0 

            for i in range(len(gates)):
                gates[i][0](gates[i][1]) 


                if index_values[i] == (len(unitary) - 1):
                    noisy_sum += noise2(gates[i][1][-2] , gates[i][1][-1]  , m*double_q_gateerror )
                else:
                    noisy_sum += noise(gates[i][1][-1] ,  m*single_q_gateerror)  


            circ.barrier(range(n))
            circ.measure(range(n), range(n))


            if noisy_sum==0:
                counts1[circs,m,r] = counts[circs,r,'ideal']

            else: 
                c = c.fromkeys(c, 0)

                result = execute(circ, simulator , shots = shots).result()
                nonzerocounts = result.get_counts(circ)

                for key in nonzerocounts:
                    c[key] = nonzerocounts[key]

                counts1[circs,m,r] =c
                
                

import pickle

with open("exp3_counts_cd=" +str(circ_depth+1)+".p", 'wb') as fp:
    pickle.dump(counts, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
with open("exp3_counts1_cd=" +str(circ_depth+1)+".p", 'wb') as fp:
    pickle.dump(counts1, fp, protocol=pickle.HIGHEST_PROTOCOL)
 
with open("exp3_correction_sum_cd=" +str(circ_depth+1)+".p", 'wb') as fp:
    pickle.dump(correction_sum, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
with open("exp3_gamma_sum_cd=" +str(circ_depth+1)+".p", 'wb') as fp:
    pickle.dump(gamma_sum, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
with open("exp3_gamma_sum2_cd=" +str(circ_depth+1)+".p", 'wb') as fp:
    pickle.dump(gamma_sum2, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    
    
    
##data processing 


job = [5,10,15,20,25,30,35,40]

mean_mit_qpr ={}
mean_mit_le={}
mean_noisy={}

stddev_mit_qpr={}
stddev_mit_le={}
stddev_noisy={}

for cd in job:
    
    
    delta_mitigated_qpr= {}
    delta_mitigated_le = {}
    delta_noisy = {}

    
    with open("exp3_counts_cd=" +str(cd)+".p", 'rb') as fp:
        counts = pickle.load(fp) 

    with open("exp3_counts1_cd=" +str(cd)+".p", 'rb') as fp:
        counts1 = pickle.load(fp)  

    with open("exp3_correction_sum_cd=" +str(cd)+".p", 'rb') as fp:
        correction_sum = pickle.load(fp)  
        
    with open("exp3_gamma_sum_cd=" +str(cd)+".p", 'rb') as fp:
        gamma_sum = pickle.load(fp)  
        
    with open("exp3_gamma_sum2_cd=" +str(cd)+".p", 'rb') as fp:
        gamma_sum2 = pickle.load(fp)  


    for circs in range(circuits): 

        idealcounts = Counter()
        for r in range (runs):
            idealcounts.update(Counter(counts[circs, r, "ideal"]))
        state = max(idealcounts.items(), key=operator.itemgetter(1))[0]


        prob_ideal={}
        prob_noisy={}
        prob_mitigated={}
        cs={}
        
        inv1 = {}
        inv2={}

        for r in range(runs):  
            prob_ideal[r] =  counts[circs, r , 'ideal'][state]/shots 
            prob_noisy[r] =  counts[circs, r , 'noisy'][state]/shots 
            prob_mitigated[r] =  (counts[circs, r , 'mitigated'][state]/shots ) *  ((-1)**(correction_sum[circs,r]))
            cs[r] = correction_sum[circs,r]
            
            inv1[r]= gamma_sum[circs,r]
            inv2[r]= gamma_sum2[circs,r]


        average_ideal = statistics.mean(prob_ideal.values())
        average_noisy = statistics.mean(prob_noisy.values())
       # average_mitigated = statistics.mean(prob_mitigated.values()) * (gamma**(sum(cs.values())))
        average_mitigated = statistics.mean(prob_mitigated.values()) * (gamma1**(sum(inv1.values()))) * (gamma2**(sum(inv2.values())))

        
        delta_mitigated_qpr[ circs] = abs(average_ideal - average_mitigated)

        delta_noisy[ circs] = abs(average_ideal - average_noisy)



        noisy_prob={}
        for m in range(2 , values+1): 
            avgr={}
            for r in range(runs):  
                avgr[r] =  counts1[circs, m , r][state] /shots   
            noisy_prob[m]= sum(avgr.values())/runs


        noisy_prob1= {1: average_noisy}
        noisy_prob = {**noisy_prob1 , **noisy_prob}

        y = list(noisy_prob.values())
        x = list(noisy_prob.keys())

        def test(x, a, b): 
            return a*x + b
        param, param_cov = curve_fit(test, x, y) #fit to error prone data 

        x1 = [0] + x
        ans = param[0]*np.array(x1) + param[1]

        delta_mitigated_le[circs] = abs(average_ideal - ans[0])
        

    
    
    mean_mit_qpr[cd] = statistics.mean(delta_mitigated_qpr.values()) 
    mean_mit_le[cd] = statistics.mean(delta_mitigated_le.values()) 
    mean_noisy[cd] = statistics.mean(delta_noisy.values()) 

    stddev_mit_qpr[cd] = statistics.stdev(delta_mitigated_qpr.values())
    stddev_mit_le[cd] = statistics.stdev(delta_mitigated_le.values())
    stddev_noisy[cd] = statistics.stdev(delta_noisy.values())

import seaborn as sns
from matplotlib.pyplot import figure
figure(num=None, figsize=(9,6), dpi=100, facecolor='w', edgecolor='k')
from pylab import *
fontsize = 15

shift = 0.0001
slide = 0.8

x = [5,10,15,20,25,30,35,40]
x1 = [i+slide for i in x]
x2 = [i-slide for i in x]


yn = list(mean_noisy.values())
yle = list(mean_mit_le.values())
yqpr = list(mean_mit_qpr.values())

ynerr = list(stddev_noisy.values())
ynerr_above = ynerr
ynerr_below = [i-shift for i in yn]


yleerr = list(stddev_mit_le.values())
yleerr_above = yleerr
yleerr_below = [i-shift for i in yleerr]


yqprerr = list(stddev_mit_qpr.values())
yqprerr_above = yqprerr
yqprerr_below = [i-shift for i in yqprerr]




# plt.plot(x, yn,'r', linestyle='dashed', linewidth = 1.5, marker='.', markersize=12, label="Noisy", ) 
# plt.plot(x, yle, 'b', linestyle='dashed', linewidth = 1.5, marker='.', markersize=12, label="LE-Mitigated") 
# plt.plot(x, yqpr,'g', linestyle='dashed', linewidth = 1.5, marker='.', markersize=12, label="PEC-Mitigated") 

plt.errorbar(x=x , y = yqpr,  yerr=(yqprerr_below, yqprerr_above ), fmt='o', color='green',ecolor='green', elinewidth=3 ,capsize=5 ,label='PEC-Mitigated')
plt.errorbar(x=x1 , y = yle, yerr=(yleerr_below ,yleerr_above ), fmt='o', color='blue',ecolor='blue', elinewidth=3, capsize=5 ,label='Mitigated LE')
plt.errorbar(x =x2, y = yn, yerr=(ynerr_below, ynerr_above), fmt='o', color='red',ecolor='red', elinewidth=3, capsize=5 ,label='Noisy')



ax = gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    
plt.xlabel('Circuit depth', fontsize=15)
plt.ylabel('Mean simulation precision', fontsize=15)
plt.legend(loc = 'best')
plt.xlim(xmin=0)  
plt.ylim(ymin=0)  
plt.savefig('exp3_noisy-errbars.png')

