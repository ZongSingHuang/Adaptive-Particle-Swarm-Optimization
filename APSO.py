# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:13:49 2020

@author: e10832
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)

class APSO():
    def __init__(self, fitness, D=30, P=20, G=500, ub=1, lb=0,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, k=0.2):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.ub = ub*np.ones([self.P, self.D])
        self.lb = lb*np.ones([self.P, self.D])
        self.w_max = w_max
        self.w_min = w_min
        self.w = w_max
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.v_max = self.k*(self.ub-self.lb)
        self.Previous_State = 'S1'
        self.rule_base = pd.DataFrame(data=[['S3', 'S2', 'S2', 'S1', 'S1', 'S1', 'S4'],
                                            ['S3', 'S2', 'S2', 'S2', 'S1', 'S1', 'S4'],
                                            ['S3', 'S3', 'S2', 'S2', 'S1', 'S4', 'S4'],
                                            ['S3', 'S3', 'S2', 'S1', 'S1', 'S4', 'S4']])
        self.rule_base.columns = ['S3', 'S3&S2', 'S2', 'S2&S1', 'S1', 'S1&S4', 'S4']
        self.rule_base.index = ['S1', 'S2', 'S3', 'S4']
        
        
        self.pbest_X = np.zeros([self.P, self.D])
        self.pbest_F = np.zeros([self.P]) + np.inf
        self.gbest_X = np.zeros([self.D])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.G)

    def opt(self):
        # 初始化
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        self.V = np.zeros([self.P, self.D])
        
        # 迭代
        for self.g in range(self.G):
            # 適應值計算
            self.F = self.fitness(self.X)
            
            # 更新最佳解
            mask = self.F < self.pbest_F
            self.pbest_X[mask] = self.X[mask].copy()
            self.pbest_F[mask] = self.F[mask].copy()
            
            if np.min(self.F) < self.gbest_F:
                idx = self.F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = self.F.min()
            
            # 收斂曲線
            self.loss_curve[self.g] = self.gbest_F
            
            # Evolutionary State Estimation
            self.EvolutionaryStateEstimation()
            
            # 更新
            r1 = np.random.uniform(size=[self.P, self.D])
            r2 = np.random.uniform(size=[self.P, self.D])
            
            self.V = self.w * self.V + self.c1 * (self.pbest_X - self.X) * r1 \
                                     + self.c2 * (self.gbest_X - self.X) * r2 # 更新V
            self.V = np.clip(self.V, -self.v_max, self.v_max) # 邊界處理
            
            self.X = self.X + self.V # 更新X
            self.X = np.clip(self.X, self.lb, self.ub) # 邊界處理

    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()
        
    def EvolutionaryStateEstimation(self):
        # step 1
        d = np.zeros([self.P])
        for i in range(self.P):
            f1 = np.sum( (self.X[i] - self.X)**2, axis=1 )
            f2 = np.sqrt( f1 )
            f3 = np.sum(f2)
            d[i] = f3/(self.P-1)
        
        # step 2
        idx = self.F.argmin()
        dmax = d.max()
        dmin = d.min()
        dg = d[idx]
        
        # step 3
        f = (dg-dmin)/(dmax-dmin)
        
        # step 4
        # Case (a)—Exploration
        if 0.0<=f<=0.4:
            uS1 = 0.0
        elif 0.4<f<=0.6:
            uS1 = 5*f - 2
        elif 0.6<f<=0.7:
            uS1 = 1.0
        elif 0.7<f<=0.8:
            uS1 = -10*f + 8
        elif 0.8<f<=1.0:
            uS1 = 0.0
        # Case (b)—Exploitation
        if 0.0<=f<=0.2:
            uS2 = 0
        elif 0.2<f<=0.3:
            uS2 = 10*f - 2
        elif 0.3<f<=0.4:
            uS2 = 1.0
        elif 0.4<f<=0.6:
            uS2 = -5*f + 3
        elif 0.6<f<=1.0:
            uS2 = 0.0
        # Case (c)—Convergence
        if 0.0<=f<=0.1:
            uS3 = 1.0
        elif 0.1<f<=0.3:
            uS3 = -5*f + 1.5
        elif 0.3<f<=1.0:
            uS3 = 0.0
        # Case (d)—Jumping Out
        if 0.0<=f<=0.7:
            uS4 = 0.0
        elif 0.7<f<=0.9:
            uS4 = 5*f - 3.5
        elif 0.9<f<=1.0:
            uS4 = 1.0
        
# =============================================================================
#      S3   S3&S2   S2   S2&S1   S1   S1&S4   S4   -> Current state
# S1   S3     S2    S2     S1    S1     S1    S4
# S2   S3     S2    S2     S2    S1     S1    S4
# S3   S3     S3    S2     S2    S1     S4    S4
# S4   S3     S3    S2     S1    S1     S4    S4
# |
# -> Previous state
# =============================================================================
        if uS3!=0:
            Current_State = 'S3'
            if uS2!=0:
                Current_State = 'S3&S2'
        elif uS2!=0:
            Current_State = 'S2'
            if uS1!=0:
                Current_State = 'S2&S1'
        elif uS1!=0:
            Current_State = 'S1'
            if uS4!=0:
                Current_State = 'S1&S4'
        elif uS4!=0:
            Current_State = 'S4'
        
        Final_State = self.rule_base[Current_State][self.Previous_State]
        self.Previous_State = Final_State

        # step 5
        delta = np.random.uniform(low=0.05, high=0.1, size=2)
        
        if Final_State=='S1': # Exploration
            self.c1 = self.c1 + delta[0]
            self.c2 = self.c2 - delta[1]
        elif Final_State=='S2': # Exploitation
            self.c1 = self.c1 + 0.5*delta[0]
            self.c2 = self.c2 - 0.5*delta[1]
        elif Final_State=='S3': # Convergence
            self.c1 = self.c1 + 0.5*delta[0]
            self.c2 = self.c2 + 0.5*delta[1]
            self.ElitistLearningStrategy()
        elif Final_State=='S4': # Jumping Out
            self.c1 = self.c1 - delta[0]
            self.c2 = self.c2 + delta[1]
            
        self.c1 = np.clip(self.c1, 1.5, 2.5)
        self.c2 = np.clip(self.c2, 1.5, 2.5)
        if (3.0<=self.c1+self.c2<=4.0)==False:
            self.c1 = 4.0 * self.c1/(self.c1+self.c2)
            self.c2 = 4.0 * self.c2/(self.c1+self.c2)

        # step 6
        self.w = 1/(1+1.5*np.exp(-2.6*f))
        self.w = np.clip(self.w, self.w_min, self.w_max)
        
    def ElitistLearningStrategy(self):
        P = self.gbest_X.copy()
        d = np.random.randint(low=0, high=self.D)
        
        mu = 0
        sigma = 1 - 0.9*self.g/self.G
        P[d] = P[d] + (self.ub[0, d]-self.lb[0, d])*np.random.normal(mu, sigma**2)
        
        P = np.clip(P, self.lb[0], self.ub[0])
        v = self.fitness(P)[0]
        
        if v<self.gbest_F:
            self.gbest_X = P.copy()
            self.gbest_F = v.copy()
        elif v<self.F.max():
            idx = self.F.argmax()
            self.X[idx] = P.copy()
            self.F[idx] = v.copy()