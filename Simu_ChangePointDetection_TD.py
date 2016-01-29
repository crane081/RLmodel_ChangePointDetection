# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 11:08:39 2015

This code is for generating stimulus sequence in a change point detection task, and 
compare how different learing rate in a TD model may lead to different behavior.

@author: chuang
"""

from scipy import stats
import numpy as np

eta = np.arange(0,1.1,.1) #learning rate range in TD model
beta = 20 #inverse decision parameter in the softmax choice model
#%% generate stimuli
n_tr = 90 #number of trials per block
n_bk = 100 #number of simulations (blocks)

#%% No change Points
xk = np.arange(3) #0,1,2
pk = (1/13,3/13,9/13) #reward probability at location 0,1,2
stimu1=np.zeros((n_tr,n_bk)) # initialization

for i in np.arange(n_bk):
    S=np.random.permutation(3) #[0 1 2] permutation 
    custm = stats.rv_discrete(name ='vs',values = (S,pk))
    stimu1[:,i] = custm.rvs(size = (n_tr)) 

#%% N(30,1) Change points occur based on N(30,1)
stimu2=np.zeros((n_tr,n_bk))
p1=np.zeros((3,n_bk)) #location of P9 for each run
L1=np.zeros((3,n_bk)) #run length
for i in np.arange(n_bk):
    L0 = np.around(np.random.normal(30,1,2)) #number of first two sections
    L=np.hstack((L0,n_tr-L0.sum())) #last one
    L1[:,i]=L
    s0=[]
    p0=[]
    for j in np.arange(len(L)): #go through each run
        S=np.random.permutation(3) #[0 1 2] permutation 
        custm = stats.rv_discrete(name = 'vs',values = (S,pk))
        a=custm.rvs(size=L[j])
        s0=np.hstack((s0,a))
        p0=np.hstack((p0,S[2])) #location of P9
        
    stimu2[:,i]=s0
    p1[:,i]=p0
    
#%% N(10,1) Change points occur based on N(10,1)
stimu3=np.zeros((n_tr,n_bk))
p2=np.zeros((9,n_bk)) #location of P9 for each run
L2=np.zeros((9,n_bk))
for i in np.arange(n_bk):
    L0 = np.around(np.random.normal(10,1,8)) #number of first 8 sections
    L=np.hstack((L0,n_tr-L0.sum())) #last one
    L2[:,i]=L
    s0=[]
    p0=[]
    for j in np.arange(len(L)):  #go through each run
        S=np.random.permutation(3) #[0 1 2] permutation 
        custm = stats.rv_discrete(name = 'vs',values = (S,pk))
        a=custm.rvs(size=L[j])
        s0=np.hstack((s0,a))
        p0=np.hstack((p0,S[2])) #location of P9

    stimu3[:,i]=s0
    p2[:,i]=p0

#%% define TD learning code (block wise)
def predict(st0,eta0): #given stimuli and eta, return model pred action
    v= np.zeros((90,3)) #initiate values
    ch=np.zeros(90) #initize model choice
    R=np.zeros(90) #initize Reward 1/0
    ch[0]=st0[0]
    R[0]=1
    for j in np.arange(89):
        if j==0:
            v[j,1]=eta0*R[0]
        else:
            v[j,ch[j]]=v[j-1,ch[j]]+eta0*(R[j]-v[j-1,ch[j]])
            #keep the other two to the same value as previous trial
            if ch[j]==0:
                v[j,1]=v[j-1,1]
                v[j,2]=v[j-1,2]
            elif ch[j]==1:
                v[j,0]=v[j-1,0]
                v[j,2]=v[j-1,2]
            elif ch[j]==2:
                v[j,0]=v[j-1,0]
                v[j,1]=v[j-1,1]
        q =np.exp(beta**v[j-1,:])/sum(np.exp(beta**v[j-1,:]))
        S = [0,1,2]
        custm = stats.rv_discrete(name ='vs',values = (S,q))
        ch[j+1] = custm.rvs()

        if ch[j+1]==st0[j+1]:
            R[j+1]=1
        
    return ch     

#%% compute P9% as a function of eta

#%% No change point
p9_eta=np.zeros((len(eta),n_bk))
for i in np.arange(len(eta)):    
    ch=np.zeros((90,n_bk))
    for j in np.arange(n_bk):
        ch[:,j] = predict(stimu1[:,j],eta[i]) #model choice
        #where P9 is at
        pp=[sum(stimu1[:,j]==0),sum(stimu1[:,j]==1),sum(stimu1[:,j]==2)]
        p9=np.argmax(pp)
        p9_eta[i,j]=sum(ch[:,j]==p9)/90

m_p9=np.mean(p9_eta,axis=1)
from scipy import stats
s_p9=stats.sem(p9_eta,axis=1)

#%%  N(30,1)
p9_eta1=np.zeros((len(eta),n_bk))
for i in np.arange(len(eta)):    
    ch=np.zeros((90,n_bk))
    for j in np.arange(n_bk):
        ch[:,j] = predict(stimu2[:,j],eta[i]) #model choice
        #where P9 is at for the entire block
        p9=np.zeros(90)
        tr=0
        for k in np.arange(3):
            for t in np.arange(L1[k,j]):
                p9[tr]=p1[k,j]
                tr=tr+1

        p9_eta1[i,j]=sum(ch[:,j]==p9)/90

m1_p9=np.mean(p9_eta1,axis=1)
from scipy import stats
s1_p9=stats.sem(p9_eta1,axis=1)
#%%  N(10,1)
p9_eta2=np.zeros((len(eta),n_bk))
for i in np.arange(len(eta)):    
    ch=np.zeros((90,n_bk))
    for j in np.arange(n_bk):
        ch[:,j] = predict(stimu3[:,j],eta[i]) #model choice
        #where P9 is at for the entire block
        p9=np.zeros(90)
        tr=0
        for k in np.arange(9):
            for t in np.arange(L2[k,j]):
                p9[tr]=p2[k,j]
                tr=tr+1

        p9_eta2[i,j]=sum(ch[:,j]==p9)/90

m2_p9=np.mean(p9_eta2,axis=1)
from scipy import stats
s2_p9=stats.sem(p9_eta2,axis=1)

#%%
import matplotlib.pyplot as plt
plt.errorbar(eta,m_p9,yerr=s_p9,label='no change point')
plt.errorbar(eta,m1_p9,yerr=s1_p9,label='N(30,1)')
plt.errorbar(eta,m2_p9,yerr=s2_p9,label='N(10,1)')

plt.xlabel('Learning rate')
plt.ylabel('P9%')
plt.title('CPD performance | learning rate')
plt.legend(loc='lowerright')
plt.show()