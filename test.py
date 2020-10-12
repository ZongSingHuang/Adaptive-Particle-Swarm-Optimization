# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 10:28:50 2020

@author: e10832
♦"""


from APSO import APSO
import numpy as np
import time
import pandas as pd
np.random.seed(42)

def Sphere(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    return np.sum(x**2, axis=1)

def Schwefel_P222(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    return np.sum(np.abs(x), axis=1) + np.prod(np.abs(x), axis=1)

def Quadric(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    outer = 0
    for i in range(x.shape[1]):
        inner = np.sum(x[:, :i+1], axis=1)**2
        outer = outer + inner
    
    return outer

def Schwefel_P221(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    fitness = np.max(np.abs(x), 1)
    
    return fitness

def Rosenbrock(x):
    if x.ndim==1:
        x = x.reshape(1, -1) 
    
    left = x[:, :-1].copy()
    right = x[:, 1:].copy()
    
    return np.sum(100*(right - left**2)**2 + (left-1)**2, axis=1)

def Step(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    return np.sum(np.round((x+0.5), 0)**2, axis=1)

def Quadric_Noise(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    matrix = np.arange(x.shape[1])+1
     
    return np.sum((x**4)*matrix, axis=1)+np.random.rand(x.shape[0])

def Schwefel(x):
    if x.ndim==1:
        x = x.reshape(1, -1)        
     
    return -1*np.sum(x*np.sin(np.abs(x)**.5), axis=1)

def Rastrigin(x):
    if x.ndim==1:
        x = x.reshape(1, -1) 
    
    return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10, axis=1)

# def Noncontinuous_Rastrigin(x):
#     if x.ndim==1:
#         x = x.reshape(1, -1)
    
#     outlier = np.abs(x)>=0.5
#     x[outlier] = np.round(2*x[outlier])/2
    
#     return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10, axis=1)

def Ackley(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    left = 20*np.exp(-0.2*(np.sum(x**2, axis=1)/x.shape[1])**.5)
    right = np.exp(np.sum(np.cos(2*np.pi*x), axis=1)/x.shape[1])
    
    return -left - right + 20 + np.e

def Griewank(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    left = np.sum(x**2, axis=1)/4000
    right = np.prod( np.cos(x/((np.arange(x.shape[1])+1)**.5)), axis=1)
    return left - right + 1

def Generalized_Penalized01(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    y_head = 1 + (x[:, 0]+1)/4
    y_tail = 1 + (x[:, -1]+1)/4
    y_left = 1 + (x[:, :-1]+1)/4
    y_right = 1 + (x[:, 1:]+1)/4
    
    first = np.pi/x.shape[1]
    second = 10*np.sin(np.pi*y_head)**2
    third = np.sum( ((y_left-1)**2) * (1+10*np.sin(np.pi*y_right)**2), axis=1)
    fourth = (y_tail-1)**2
    five = np.sum(u_xakm(x, 10, 100, 4), axis=1)

    fitness = first*(second + third + fourth) + five
    
    return fitness

def Generalized_Penalized02(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    x_head = x[:, 0]
    x_tail = x[:, -1]
    x_left = x[:, :-1]
    x_right = x[:, 1:]
       
    first = 0.1
    second = np.sin(3*np.pi*x_head)**2 + (x_tail-1)**2
    third = np.sum( (x_left-1)**2 * (1+np.sin(3*np.pi*x_right)**2), axis=1)
    fourth = np.sum(u_xakm(x, 5, 100, 4), axis=1)

    fitness = first*(second + third) + fourth
    
    return fitness

def DE_JONG_N5(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    a1 = np.array([-32, -16, 0, 16, 32, 
                   -32, -16, 0, 16, 32, 
                   -32, -16, 0, 16, 32, 
                   -32, -16, 0, 16, 32, 
                   -32, -16, 0, 16, 32])
    a2 = np.array([-32, -32, -32, -32, -32,
                   -16, -16, -16, -16, -16,
                     0,   0,   0,   0,   0,
                    16,  16,  16,  16,  16,
                    32,  32,  32,  32,  32])
    
    fitness = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        x1 = x[i, 0]
        x2 = x[i, 1]
        
        term1 = np.arange(25)+1
        term2 = (x1-a1)**6
        term3 = (x2-a2)**6
        term_left = np.sum(1/(term1 + term2 + term3))
        term_right = 1/500
        
        fitness[i] = 1/(term_right + term_left)
    
    return fitness

def Kowalik(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    aK = np.array([.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246])
    bK = 1/np.array([.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16])

    fitness = np.zeros(x.shape[0])
    for i in range(x.shape[0]):    
        term1 = x[i, 0]*(bK**2+x[i, 1]*bK)
        term2 = bK**2+x[i, 2]*bK+x[i, 3]       
        fitness[i] = np.sum((aK - term1/term2)**2)
    
    return fitness

def Six_Hump_Camel(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    return 4*(x[:, 0]**2)-2.1*(x[:, 0]**4)+(x[:, 0]**6)/3+x[:, 0]*x[:, 1]-4*(x[:, 1]**2)+4*(x[:, 1]**4)

def Brain(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    return (x[:, 1]-(x[:, 0]**2)*5.1/(4*(np.pi**2))+5/np.pi*x[:, 0]-6)**2+10*(1-1/(8*np.pi))*np.cos(x[:, 0])+10

def Goldstein_Price(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    return (1+(x[:, 0]+x[:, 1]+1)**2*(19-14*x[:, 0]+3*(x[:, 0]**2)-14*x[:, 1]+6*x[:, 0]*x[:, 1]+3*x[:, 1]**2))* \
    (30+(2*x[:, 0]-3*x[:, 1])**2*(18-32*x[:, 0]+12*(x[:, 0]**2)+48*x[:, 1]-36*x[:, 0]*x[:, 1]+27*(x[:, 1]**2)))

def Hartmann_3D(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    aH = np.array([[3, 10, 30],
                   [.1, 10, 35],
                   [3, 10, 30],
                   [.1, 10, 35]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[.3689, .117, .2673],
                   [.4699, .4387, .747],
                   [.1091, .8732, .5547],
                   [.03815, .5743, .8828]])
    
    fitness = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        term1 = cH[0]*np.exp( -1*np.sum( aH[0, :]*(x[i, :]-pH[0, :])**2, axis=0 ) )
        term2 = cH[1]*np.exp( -1*np.sum( aH[1, :]*(x[i, :]-pH[1, :])**2, axis=0 ) )
        term3 = cH[2]*np.exp( -1*np.sum( aH[2, :]*(x[i, :]-pH[2, :])**2, axis=0 ) )
        term4 = cH[3]*np.exp( -1*np.sum( aH[3, :]*(x[i, :]-pH[3, :])**2, axis=0 ) )
        
        fitness[i] = -1*(term1 + term2 + term3 + term4)
    
    return fitness

def Hartmann_6D(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    aH = np.array([[10, 3, 17, 3.5, 1.7, 8],
                   [.05, 10, 17, .1, 8, 14],
                   [3, 3.5, 1.7, 10, 17, 8],
                   [17, 8, .05, 10, .1, 14]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[.1312, .1696, .5569, .0124, .8283, .5886],
                   [.2329, .4135, .8307, .3736, .1004, .9991],
                   [.2348, .1415, .3522, .2883, .3047, .6650],
                   [.4047, .8828, .8732, .5743, .1091, .0381]])
    
    fitness = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        term1 = cH[0]*np.exp( -1*np.sum( aH[0, :]*(x[i, :]-pH[0, :])**2, axis=0 ) )
        term2 = cH[1]*np.exp( -1*np.sum( aH[1, :]*(x[i, :]-pH[1, :])**2, axis=0 ) )
        term3 = cH[2]*np.exp( -1*np.sum( aH[2, :]*(x[i, :]-pH[2, :])**2, axis=0 ) )
        term4 = cH[3]*np.exp( -1*np.sum( aH[3, :]*(x[i, :]-pH[3, :])**2, axis=0 ) )
        
        fitness[i] = -1*(term1 + term2 + term3 + term4)
    
    return fitness

def Shekel_m5(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    aSH = np.array([[4, 4, 4, 4], 
                    [1, 1, 1, 1], 
                    [8, 8, 8, 8],
                    [6, 6, 6, 6],
                    [3, 7, 3, 7],
                    [2, 9, 2, 9],
                    [5, 5, 3, 3],
                    [8, 1, 8, 1],
                    [6, 2, 6, 2],
                    [7, 3.6, 7, 3.6]])
    cSH=np.array([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])
    
    fitness = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(5):
            fitness[i] = fitness[i] - 1/(np.dot((x[i, :]-aSH[j, :]), (x[i, :]-aSH[j,:]).T)+cSH[j])
    return fitness


def Shekel_m7(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    aSH = np.array([[4, 4, 4, 4], 
                    [1, 1, 1, 1], 
                    [8, 8, 8, 8],
                    [6, 6, 6, 6],
                    [3, 7, 3, 7],
                    [2, 9, 2, 9],
                    [5, 5, 3, 3],
                    [8, 1, 8, 1],
                    [6, 2, 6, 2],
                    [7, 3.6, 7, 3.6]])
    cSH=np.array([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])
    
    fitness = np.zeros(x.shape[0])
    for i in range(x.shape[0]):    
        for j in range(7):
            fitness[i] = fitness[i] - 1/(np.dot((x[i, :]-aSH[j, :]), (x[i, :]-aSH[j,:]).T)+cSH[j])
    return fitness

def Shekel_m10(x):
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    aSH = np.array([[4, 4, 4, 4], 
                    [1, 1, 1, 1], 
                    [8, 8, 8, 8],
                    [6, 6, 6, 6],
                    [3, 7, 3, 7],
                    [2, 9, 2, 9],
                    [5, 5, 3, 3],
                    [8, 1, 8, 1],
                    [6, 2, 6, 2],
                    [7, 3.6, 7, 3.6]])
    cSH=np.array([.1, .2, .2, .4, .4, .6, .3, .7, .5, .5])

    fitness = np.zeros(x.shape[0])
    for i in range(x.shape[0]):  
        for j in range(10):
            fitness[i] = fitness[i] - 1/(np.dot((x[i, :]-aSH[j, :]), (x[i, :]-aSH[j,:]).T)+cSH[j])
    return fitness

def u_xakm(x, a, k, m):
    if x.ndim==1:
        x = x.reshape(1, -1)
    temp = x.copy()    
    
    case1 = x>a
    case3 = x<-a
    
    temp = np.zeros_like(x)
    temp[case1] = k*(x[case1]-a)**m         
    temp[case3] = k*(-1*x[case3]-a)**m
    
    return temp


d = 30
g = 500
p = 30
times = 50
table = np.zeros((5, 23)) # ['avg', 'time', 'worst', 'best', 'std']
table[2, :] = -np.ones(23)*np.inf # worst
table[3, :] = np.ones(23)*np.inf # best
all_for_std = np.zeros((times, 23))
all_for_loss = np.zeros((g, 23))
for i in range(times):
    x_max = 100*np.ones(d)
    x_min = -100*np.ones(d)
    optimizer = APSO(fit_func=Sphere,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 0]: table[2, 0] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 0]: table[3, 0] = optimizer.gBest_score
    table[0, 0] += optimizer.gBest_score
    table[1, 0] += end - start 
    all_for_std[i, 0] = optimizer.gBest_score
    all_for_loss[:, 0] += optimizer.gBest_curve


    x_max = 10*np.ones(d)
    x_min = -10*np.ones(d)
    optimizer = APSO(fit_func=Schwefel_P222,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 1]: table[2, 1] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 1]: table[3, 1] = optimizer.gBest_score
    table[0, 1] += optimizer.gBest_score
    table[1, 1] += end - start  
    all_for_std[i, 1] = optimizer.gBest_score
    all_for_loss[:, 1] += optimizer.gBest_curve

    
    x_max = 100*np.ones(d)
    x_min = -100*np.ones(d)
    optimizer = APSO(fit_func=Quadric,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 2]: table[2, 2] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 2]: table[3, 2] = optimizer.gBest_score
    table[0, 2] += optimizer.gBest_score
    table[1, 2] += end - start
    all_for_std[i, 2] = optimizer.gBest_score
    all_for_loss[:, 2] += optimizer.gBest_curve

    x_max = 100*np.ones(d)
    x_min = -100*np.ones(d)
    optimizer = APSO(fit_func=Schwefel_P221,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 3]: table[2, 3] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 3]: table[3, 3] = optimizer.gBest_score  
    table[0, 3] += optimizer.gBest_score
    table[1, 3] += end - start  
    all_for_std[i, 3] = optimizer.gBest_score
    all_for_loss[:, 3] += optimizer.gBest_curve
 
    x_max = 30*np.ones(d)
    x_min = -30*np.ones(d)
    optimizer = APSO(fit_func=Rosenbrock,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 4]: table[2, 4] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 4]: table[3, 4] = optimizer.gBest_score  
    table[0, 4] += optimizer.gBest_score
    table[1, 4] += end - start  
    all_for_std[i, 4] = optimizer.gBest_score
    all_for_loss[:, 4] += optimizer.gBest_curve

   
    x_max = 100*np.ones(d)
    x_min = -100*np.ones(d)
    optimizer = APSO(fit_func=Step,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 5]: table[2, 5] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 5]: table[3, 5] = optimizer.gBest_score  
    table[0, 5] += optimizer.gBest_score
    table[1, 5] += end - start
    all_for_std[i, 5] = optimizer.gBest_score
    all_for_loss[:, 5] += optimizer.gBest_curve
  
  
    x_max = 1.28*np.ones(d)
    x_min = -1.28*np.ones(d)
    optimizer = APSO(fit_func=Quadric_Noise,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 6]: table[2, 6] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 6]: table[3, 6] = optimizer.gBest_score   
    table[0, 6] += optimizer.gBest_score
    table[1, 6] += end - start
    all_for_std[i, 6] = optimizer.gBest_score
    all_for_loss[:, 6] += optimizer.gBest_curve
 
 
    x_max = 500*np.ones(d)
    x_min = -500*np.ones(d)
    optimizer = APSO(fit_func=Schwefel,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 7]: table[2, 7] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 7]: table[3, 7] = optimizer.gBest_score   
    table[0, 7] += optimizer.gBest_score
    table[1, 7] += end - start
    all_for_std[i, 7] = optimizer.gBest_score
    all_for_loss[:, 7] += optimizer.gBest_curve
  

    x_max = 5.12*np.ones(d)
    x_min = -5.12*np.ones(d)
    optimizer = APSO(fit_func=Rastrigin,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 8]: table[2, 8] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 8]: table[3, 8] = optimizer.gBest_score   
    table[0, 8] += optimizer.gBest_score
    table[1, 8] += end - start  
    all_for_std[i, 8] = optimizer.gBest_score
    all_for_loss[:, 8] += optimizer.gBest_curve 
 
    x_max = 32*np.ones(d)
    x_min = -32*np.ones(d)
    optimizer = APSO(fit_func=Ackley,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 9]: table[2, 9] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 9]: table[3, 9] = optimizer.gBest_score  
    table[0, 9] += optimizer.gBest_score
    table[1, 9] += end - start
    all_for_std[i, 9] = optimizer.gBest_score
    all_for_loss[:, 9] += optimizer.gBest_curve
   
 
    x_max = 600*np.ones(d)
    x_min = -600*np.ones(d)
    optimizer = APSO(fit_func=Griewank,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 10]: table[2, 10] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 10]: table[3, 10] = optimizer.gBest_score  
    table[0, 10] += optimizer.gBest_score
    table[1, 10] += end - start  
    all_for_std[i, 10] = optimizer.gBest_score
    all_for_loss[:, 10] += optimizer.gBest_curve

    x_max = 50*np.ones(d)
    x_min = -50*np.ones(d)
    optimizer = APSO(fit_func=Generalized_Penalized01,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 11]: table[2, 11] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 11]: table[3, 11] = optimizer.gBest_score  
    table[0, 11] += optimizer.gBest_score
    table[1, 11] += end - start  
    all_for_std[i, 11] = optimizer.gBest_score
    all_for_loss[:, 11] += optimizer.gBest_curve
    
    x_max = 50*np.ones(d)
    x_min = -50*np.ones(d)
    optimizer = APSO(fit_func=Generalized_Penalized02,
                    num_dim=d, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 12]: table[2, 12] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 12]: table[3, 12] = optimizer.gBest_score  
    table[0, 12] += optimizer.gBest_score
    table[1, 12] += end - start  
    all_for_std[i, 12] = optimizer.gBest_score
    all_for_loss[:, 12] += optimizer.gBest_curve
    
    x_max = 65.536*np.ones(2)
    x_min = -65.536*np.ones(2)
    optimizer = APSO(fit_func=DE_JONG_N5,
                    num_dim=2, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 13]: table[2, 13] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 13]: table[3, 13] = optimizer.gBest_score  
    table[0, 13] += optimizer.gBest_score
    table[1, 13] += end - start  
    all_for_std[i, 13] = optimizer.gBest_score
    all_for_loss[:, 13] += optimizer.gBest_curve
    
    x_max = 5*np.ones(4)
    x_min = -5*np.ones(4)
    optimizer = APSO(fit_func=Kowalik,
                    num_dim=4, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 14]: table[2, 14] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 14]: table[3, 14] = optimizer.gBest_score  
    table[0, 14] += optimizer.gBest_score
    table[1, 14] += end - start  
    all_for_std[i, 14] = optimizer.gBest_score
    all_for_loss[:, 14] += optimizer.gBest_curve
    
    x_max = 5*np.ones(2)
    x_min = -5*np.ones(2)
    optimizer = APSO(fit_func=Six_Hump_Camel,
                    num_dim=2, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 15]: table[2, 15] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 15]: table[3, 15] = optimizer.gBest_score  
    table[0, 15] += optimizer.gBest_score
    table[1, 15] += end - start  
    all_for_std[i, 15] = optimizer.gBest_score
    all_for_loss[:, 15] += optimizer.gBest_curve
    
    x_max = 5*np.ones(2)
    x_min = -5*np.ones(2)
    optimizer = APSO(fit_func=Brain,
                    num_dim=2, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 16]: table[2, 16] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 16]: table[3, 16] = optimizer.gBest_score  
    table[0, 16] += optimizer.gBest_score
    table[1, 16] += end - start  
    all_for_std[i, 16] = optimizer.gBest_score
    all_for_loss[:, 16] += optimizer.gBest_curve
    
    x_max = 2*np.ones(2)
    x_min = -2*np.ones(2)
    optimizer = APSO(fit_func=Goldstein_Price,
                    num_dim=2, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 17]: table[2, 17] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 17]: table[3, 17] = optimizer.gBest_score  
    table[0, 17] += optimizer.gBest_score
    table[1, 17] += end - start  
    all_for_std[i, 17] = optimizer.gBest_score
    all_for_loss[:, 17] += optimizer.gBest_curve
    
    x_max = 1*np.ones(3)
    x_min = 0*np.ones(3)
    optimizer = APSO(fit_func=Hartmann_3D,
                    num_dim=3, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 18]: table[2, 18] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 18]: table[3, 18] = optimizer.gBest_score  
    table[0, 18] += optimizer.gBest_score
    table[1, 18] += end - start  
    all_for_std[i, 18] = optimizer.gBest_score
    all_for_loss[:, 18] += optimizer.gBest_curve
    
    x_max = 1*np.ones(6)
    x_min = 0*np.ones(6)
    optimizer = APSO(fit_func=Hartmann_6D,
                    num_dim=6, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 19]: table[2, 19] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 19]: table[3, 19] = optimizer.gBest_score  
    table[0, 19] += optimizer.gBest_score
    table[1, 19] += end - start  
    all_for_std[i, 19] = optimizer.gBest_score
    all_for_loss[:, 19] += optimizer.gBest_curve
    
    x_max = 10*np.ones(4)
    x_min = 0*np.ones(4)
    optimizer = APSO(fit_func=Shekel_m5,
                    num_dim=4, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 20]: table[2, 20] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 20]: table[3, 20] = optimizer.gBest_score  
    table[0, 20] += optimizer.gBest_score
    table[1, 20] += end - start  
    all_for_std[i, 20] = optimizer.gBest_score
    all_for_loss[:, 20] += optimizer.gBest_curve
    
    x_max = 10*np.ones(4)
    x_min = 0*np.ones(4)
    optimizer = APSO(fit_func=Shekel_m7,
                    num_dim=4, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 21]: table[2, 21] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 21]: table[3, 21] = optimizer.gBest_score  
    table[0, 21] += optimizer.gBest_score
    table[1, 21] += end - start  
    all_for_std[i, 21] = optimizer.gBest_score
    all_for_loss[:, 21] += optimizer.gBest_curve
    
    x_max = 10*np.ones(4)
    x_min = 0*np.ones(4)
    optimizer = APSO(fit_func=Shekel_m10,
                    num_dim=4, num_particle=p, max_iter=g, x_max=x_max, x_min=x_min)
    start = time.time()
    optimizer.opt()
    end = time.time()
    if optimizer.gBest_score>table[2, 22]: table[2, 22] = optimizer.gBest_score
    if optimizer.gBest_score<table[3, 22]: table[3, 22] = optimizer.gBest_score  
    table[0, 22] += optimizer.gBest_score
    table[1, 22] += end - start  
    all_for_std[i, 22] = optimizer.gBest_score
    all_for_loss[:, 22] += optimizer.gBest_curve
    
    print(i+1)
    
    
table[:2, :] = table[:2, :] / times
table[4, :] = np.std(all_for_std, axis=0)
table = pd.DataFrame(table)
table.columns=['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 
                'F8', 'F9', 'F10', 'F11', 'F12', 
                'F13', 'F14', 'F15', 'F16', 'F17', 'F18',
                'F19', 'F20', 'F21', 'F22', 'F23']
table.index = ['avg', 'time', 'worst', 'best', 'std']


all_for_loss = all_for_loss / times
all_for_loss = pd.DataFrame(all_for_loss)
all_for_loss.columns=['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 
                      'F8', 'F9', 'F10', 'F11', 'F12', 
                      'F13', 'F14', 'F15', 'F16', 'F17', 'F18',
                      'F19', 'F20', 'F21', 'F22', 'F23']
ax = all_for_loss.plot(kind='line', grid=True, legend=True, logy=True)
ax.set_title('WOA')
ax.set_xlabel('iteration')
ax.set_ylabel('fitness value')