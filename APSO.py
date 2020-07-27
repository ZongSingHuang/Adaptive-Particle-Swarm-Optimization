# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:03:40 2020

@author: e10832
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

class APSO():
    def __init__(self, fit_func, num_dim, num_particle=20, max_iter=500,
                 x_max=1, x_min=-1, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, k=0.2):
        self.fit_func = fit_func        
        self.num_dim = num_dim
        self.num_particle = num_particle
        self.max_iter = max_iter
        
        self.x_max = x_max
        self.x_min = x_min
        self.w = w_max        
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self._iter = 1
        self.bestX_idx = -1
        self.worstX_idx = -1
        # case1. 僅初始化
        self.delta = np.random.uniform(high=0.1, low=0.05, size=1)
        self.state = 1
        self.rulebase = [[1, 1, 4, 1], 
                         [2, 2, 2, 1], 
                         [2, 3, 3, 3], 
                         [4, 3, 4, 4]]
        self.gBest_curve = np.zeros(self.max_iter)
 
        self.X = np.random.uniform(size=[self.num_particle, self.num_dim])*(self.x_max-self.x_min) + self.x_min
        self.V = np.random.uniform(size=[self.num_particle, self.num_dim])*(self.x_max-self.x_min) + self.x_min
        self.v_max = self.k*(self.x_max-self.x_min)
        
        self.pBest_X = self.X.copy()
        self.pBest_score = self.fit_func(self.X).copy()
        self.X_score = self.pBest_score.copy()
        self.gBest_X = self.pBest_X[self.pBest_score.argmin()].copy()
        self.gBest_score = self.pBest_score.min().copy()
        self.gBest_curve[0] = self.gBest_score.copy()
        self.bestX_idx = self.pBest_score.argmin()
        self.worstX_idx = self.pBest_score.argmax()

    def ESE(self):
        # step 1.
        d_set = np.zeros(self.num_particle)
                      
        for i in range(self.num_particle):
            d_set[i] = np.sum(np.linalg.norm((self.X[i, :].reshape(1, self.num_dim)-self.X), ord=2, axis=1))\
                /(self.num_particle-1)
        
        # step 2.        
        dmax = np.max(d_set)
        dmin = np.min(d_set)
        dg = d_set[self.bestX_idx].copy()        
        # 我自己額外加的，防止崩潰
        if dmax==dmin==dg==0:
            f = 0
        else:
            f = (dg-dmin)/(dmax-dmin)

        if not(0<=f<=1):
            print('f='+str(np.round(f, 3))+' 超出範圍[0, 1]!!!')            
            if f>1:
                f = 1
            elif f<0:
                f = 0 
            
        # step 3.
        singleton = np.zeros(4)
        if 0<=f<=0.4:
            singleton[0] = 0
        elif 0.4<f<=0.6:
            singleton[0] = 5*f - 2
        elif 0.6<f<=0.7:
            singleton[0] = 1
        elif 0.7<f<=0.8:
            singleton[0] = -10*f + 8
        elif 0.8<f<=1:
            singleton[0] = 0

        if 0<=f<=0.2:
            singleton[1] = 0
        elif 0.2<f<=0.3:
            singleton[1] = 10*f - 2
        elif 0.3<f<=0.4:
            singleton[1] = 1
        elif 0.4<f<=0.6:
            singleton[1] = -5*f + 3
        elif 0.6<f<=1:
            singleton[1] = 0

        if 0<=f<=0.1:
            singleton[2] = 1
        elif 0.1<f<=0.3:
            singleton[2] = -5*f + 1.5
        elif 0.3<f<=1:
            singleton[2] = 0
            
        if 0<=f<=0.7:
            singleton[3] = 0
        elif 0.7<f<=0.9:
            singleton[3] = 5*f - 3.5
        elif 0.9<f<=1:
            singleton[3] = 1
        
        if np.max(singleton)==np.min(singleton)==0:
            print('因為f超出範圍[0, 1]，所以模糊推論異常!!!')
        
        t = np.argmax(singleton)
        t_1 = self.state - 1
        self.state = self.rulebase[int(t)][int(t_1)]
    
        # (10)
        self.w = 1/(1+1.5*np.exp(-2.6*f))
        if not(self.w_min<=self.w<=self.w_max):
            print('w='+str(np.round(self.w, 3))+' 超出範圍[0.4, 0.9]!!!')
            if self.w<self.w_min:
                self.w = self.w_min
            elif self.w>self.w_max:
                self.w = self.w_max
                   
        # (11)
        if self.state==1:
            self.c1 = self.c1 + self.delta
            self.c2 = self.c2 - self.delta
        if self.state==2:
            self.c1 = self.c1 + 0.5*self.delta
            self.c2 = self.c2 - 0.5*self.delta
        if self.state==3:
            self.c1 = self.c1 + 0.5*self.delta
            self.c2 = self.c2 + 0.5*self.delta
            self.ELS()
        if self.state==4:
            self.c1 = self.c1 - self.delta
            self.c2 = self.c2 + self.delta
        if not(1.5<=self.c1<=2.5):
            # print('c1='+str(np.round(self.c1, 3))+' 超出範圍[1.5, 2.5]!!!') 
            if self.c1<1.5:
                self.c1 = 1.5
            elif self.c1>2.5:
                self.c1 = 2.5
        if not(1.5<=self.c2<=2.5):
            # print('c2='+str(np.round(self.c2, 3))+' 超出範圍[1.5, 2.5]!!!')  
            if self.c2<1.5:
                self.c2 = 1.5
            elif self.c2>2.5:
                self.c2 = 2.5
            
        # (12)
        if not(3<=self.c1+self.c2<=4):
            # print('c1='+str(np.round(self.c1, 3))+' + c2='+str(np.round(self.c2, 3))+' 超出範圍[3, 4]!!!')
            self.c1, self.c2 = 4.0*self.c1/(self.c1+self.c2), 4.0*self.c2/(self.c1+self.c2)            
    def ELS(self):
        rho = 1 - ((1-0.1)*self._iter/self.max_iter)
        d = np.random.randint(low=0, high=self.num_dim, size=1)
        P = self.gBest_X.copy()
        P[d] = P[d] + (self.x_max[d]-self.x_min[d])*np.random.normal(loc=0.0, scale=rho**2, size=1)
        
        if P[d]>self.x_max[d]: P[d] = self.x_max[d]
        if P[d]<self.x_min[d]: P[d] = self.x_min[d]
        
        score = self.fit_func(P.reshape(1, -1))
        
        if score<self.gBest_score:
            self.gBest_X = P.copy()
            self.gBest_score = score.copy()
        else: 
            # # case1. 原文版本
            # self.X[self.worstX_idx, :] = P.copy()
            
            # case2. 我的版本
            if score<self.X_score[self.worstX_idx]:
                self.X[self.worstX_idx, :] = P.copy()
            if score<np.max(self.pBest_score):
                idx = np.argmax(self.pBest_score)
                self.pBest_score[idx] = score.copy()
                self.pBest_X[idx, :] = P.copy()  

    def opt(self):
        while(self._iter<self.max_iter):   
            # case2. 每代更新
            # self.delta = np.random.uniform(high=0.1, low=0.05, size=1)
            self.bestX_idx = self.X_score.argmin()
            self.worstX_idx = self.X_score.argmax()
            
            self.ESE()
            R1 = np.random.uniform(size=(self.num_particle, self.num_dim))
            R2 = np.random.uniform(size=(self.num_particle, self.num_dim))
            for i in range(self.num_particle):                              
                self.V[i, :] = self.w * self.V[i, :] \
                        + self.c1 * (self.pBest_X[i, :] - self.X[i, :]) * R1[i, :] \
                        + self.c2 * (self.gBest_X - self.X[i, :]) * R2[i, :]               
                self.V[i, self.v_max < self.V[i, :]] = self.v_max[self.v_max < self.V[i, :]]
                self.V[i, -self.v_max > self.V[i, :]] = -self.v_max[-self.v_max > self.V[i, :]]
                
                self.X[i, :] = self.X[i, :] + self.V[i, :]
                self.X[i, self.x_max < self.X[i, :]] = self.x_max[self.x_max < self.X[i, :]]
                self.X[i, self.x_min > self.X[i, :]] = self.x_min[self.x_min > self.X[i, :]]
                
                score = self.fit_func(self.X[i, :])
                self.X_score[i] = score.copy()
                if score<self.pBest_score[i]:
                    self.pBest_X[i, :] = self.X[i, :].copy()
                    self.pBest_score[i] = score.copy()
                    if score<self.gBest_score:                        
                        self.gBest_X = self.X[i, :].copy()
                        self.gBest_score = score.copy()
                                       
            self.gBest_curve[self._iter] = self.gBest_score.copy()         
            self._iter = self._iter + 1            


    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()      

    