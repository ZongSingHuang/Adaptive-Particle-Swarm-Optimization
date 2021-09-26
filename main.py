# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 21:43:03 2020

@author: ZongSing_NB
"""

import time
import functools

import numpy as np
import pandas as pd

from APSO import APSO
import benchmark
import bound_X
import ideal_F
import dimension

D = 30
G = 500
P = 30
run_times = 50
table = pd.DataFrame(np.zeros([6, 36]), index=['avg', 'std', 'worst', 'best', 'ideal', 'time'])
loss_curves = np.zeros([G, 36])
F_table = np.zeros([run_times, 36])
for t in range(run_times):
    item = 0
    ub = bound_X.Sphere()[1]*np.ones(dimension.Sphere(D))
    lb = bound_X.Sphere()[0]*np.ones(dimension.Sphere(D))
    optimizer = APSO(fitness=benchmark.Sphere,
                     D=dimension.Sphere(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Sphere()
    loss_curves[:, item] += optimizer.loss_curve

    
    item = item + 1
    ub = bound_X.Rastrigin()[1]*np.ones(dimension.Rastrigin(D))
    lb = bound_X.Rastrigin()[0]*np.ones(dimension.Rastrigin(D))
    optimizer = APSO(fitness=benchmark.Rastrigin,
                      D=dimension.Rastrigin(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Rastrigin()
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = bound_X.Ackley()[1]*np.ones(dimension.Ackley(D))
    lb = bound_X.Ackley()[0]*np.ones(dimension.Ackley(D))
    optimizer = APSO(fitness=benchmark.Ackley,
                      D=dimension.Ackley(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Ackley()
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = bound_X.Griewank()[1]*np.ones(dimension.Griewank(D))
    lb = bound_X.Griewank()[0]*np.ones(dimension.Griewank(D))
    optimizer = APSO(fitness=benchmark.Griewank,
                      D=dimension.Griewank(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Griewank()
    loss_curves[:, item] += optimizer.loss_curve
    

    item = item + 1
    ub = bound_X.Schwefel_P222()[1]*np.pi*np.ones(dimension.Schwefel_P222(D))
    lb = bound_X.Schwefel_P222()[0]*np.pi*np.ones(dimension.Schwefel_P222(D))
    optimizer = APSO(fitness=benchmark.Schwefel_P222,
                      D=dimension.Schwefel_P222(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Schwefel_P222()
    loss_curves[:, item] += optimizer.loss_curve
    

    item = item + 1
    ub = bound_X.Rosenbrock()[1]*np.ones(dimension.Rosenbrock(D))
    lb = bound_X.Rosenbrock()[0]*np.ones(dimension.Rosenbrock(D))
    optimizer = APSO(fitness=benchmark.Rosenbrock,
                      D=dimension.Rosenbrock(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Rosenbrock()
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = bound_X.Sehwwefel_P221()[1]*np.ones(dimension.Sehwwefel_P221(D))
    lb = bound_X.Sehwwefel_P221()[0]*np.ones(dimension.Sehwwefel_P221(D))
    optimizer = APSO(fitness=benchmark.Sehwwefel_P221,
                      D=dimension.Sehwwefel_P221(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Sehwwefel_P221()
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = bound_X.Quartic()[1]*np.ones(dimension.Quartic(D))
    lb = bound_X.Quartic()[0]*np.ones(dimension.Quartic(D))
    optimizer = APSO(fitness=benchmark.Quartic,
                      D=dimension.Quartic(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Quartic()
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = bound_X.Schwefel_P12()[1]*np.ones(dimension.Schwefel_P12(D))
    lb = bound_X.Schwefel_P12()[0]*np.ones(dimension.Schwefel_P12(D))
    optimizer = APSO(fitness=benchmark.Schwefel_P12,
                      D=dimension.Schwefel_P12(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Schwefel_P12()
    loss_curves[:, item] += optimizer.loss_curve


    item = item + 1
    ub = bound_X.Penalized1()[1]*np.ones(dimension.Penalized1(D))
    lb = bound_X.Penalized1()[0]*np.ones(dimension.Penalized1(D))
    optimizer = APSO(fitness=benchmark.Penalized1,
                      D=dimension.Penalized1(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Penalized1()
    loss_curves[:, item] += optimizer.loss_curve
    

    item = item + 1
    ub = bound_X.Penalized2()[1]*np.ones(dimension.Penalized2(D))
    lb = bound_X.Penalized2()[0]*np.ones(dimension.Penalized2(D))
    optimizer = APSO(fitness=benchmark.Penalized2,
                      D=dimension.Penalized2(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Penalized2()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Schwefel_226()[1]*np.ones(dimension.Schwefel_226(D))
    lb = bound_X.Schwefel_226()[0]*np.ones(dimension.Schwefel_226(D))
    optimizer = APSO(fitness=benchmark.Schwefel_226,
                      D=dimension.Schwefel_226(dimension.Schwefel_226(D)), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Schwefel_226(D)
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Step()[1]*np.ones(dimension.Step(D))
    lb = bound_X.Step()[0]*np.ones(dimension.Step(D))
    optimizer = APSO(fitness=benchmark.Step,
                      D=dimension.Step(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Step()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Kowalik()[1]*np.ones(dimension.Kowalik())
    lb = bound_X.Kowalik()[0]*np.ones(dimension.Kowalik())
    optimizer = APSO(fitness=benchmark.Kowalik,
                      D=dimension.Kowalik(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Kowalik()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.ShekelFoxholes()[1]*np.ones(dimension.ShekelFoxholes())
    lb = bound_X.ShekelFoxholes()[0]*np.ones(dimension.ShekelFoxholes())
    optimizer = APSO(fitness=benchmark.ShekelFoxholes,
                      D=dimension.ShekelFoxholes(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.ShekelFoxholes()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.GoldsteinPrice()[1]*np.ones(dimension.GoldsteinPrice())
    lb = bound_X.GoldsteinPrice()[0]*np.ones(dimension.GoldsteinPrice())
    optimizer = APSO(fitness=benchmark.GoldsteinPrice,
                      D=dimension.GoldsteinPrice(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.GoldsteinPrice()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    Shekel5 = functools.partial(benchmark.Shekel, m=5)
    ub = bound_X.Shekel()[1]*np.ones(dimension.Shekel())
    lb = bound_X.Shekel()[0]*np.ones(dimension.Shekel())
    optimizer = APSO(fitness=Shekel5,
                      D=dimension.Shekel(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Shekel()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Branin()[2:]*np.ones(dimension.Branin())
    lb = bound_X.Branin()[:2]*np.ones(dimension.Branin())
    optimizer = APSO(fitness=benchmark.Branin,
                      D=dimension.Branin(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Branin()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Hartmann3()[1]*np.ones(dimension.Hartmann3())
    lb = bound_X.Hartmann3()[0]*np.ones(dimension.Hartmann3())
    optimizer = APSO(fitness=benchmark.Hartmann3,
                      D=dimension.Hartmann3(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Hartmann3()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    Shekel7 = functools.partial(benchmark.Shekel, m=7)
    ub = bound_X.Shekel()[1]*np.ones(dimension.Shekel())
    lb = bound_X.Shekel()[0]*np.ones(dimension.Shekel())
    optimizer = APSO(fitness=Shekel7,
                      D=dimension.Shekel(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Shekel()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    Shekel10 = functools.partial(benchmark.Shekel, m=10)
    ub = bound_X.Shekel()[1]*np.ones(dimension.Shekel())
    lb = bound_X.Shekel()[0]*np.ones(dimension.Shekel())
    optimizer = APSO(fitness=benchmark.Shekel,
                      D=dimension.Shekel(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Shekel()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.SixHumpCamelBack()[1]*np.ones(dimension.SixHumpCamelBack())
    lb = bound_X.SixHumpCamelBack()[0]*np.ones(dimension.SixHumpCamelBack())
    optimizer = APSO(fitness=benchmark.SixHumpCamelBack,
                      D=dimension.SixHumpCamelBack(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.SixHumpCamelBack()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Hartmann6()[1]*np.ones(dimension.Hartmann6())
    lb = bound_X.Hartmann6()[0]*np.ones(dimension.Hartmann6())
    optimizer = APSO(fitness=benchmark.Hartmann6,
                      D=dimension.Hartmann6(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Hartmann6()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Zakharov()[1]*np.ones(dimension.Zakharov(D))
    lb = bound_X.Zakharov()[0]*np.ones(dimension.Zakharov(D))
    optimizer = APSO(fitness=benchmark.Zakharov,
                      D=dimension.Zakharov(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Zakharov()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.SumSquares()[1]*np.ones(dimension.SumSquares(D))
    lb = bound_X.SumSquares()[0]*np.ones(dimension.SumSquares(D))
    optimizer = APSO(fitness=benchmark.SumSquares,
                      D=dimension.SumSquares(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.SumSquares()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Alpine()[1]*np.ones(dimension.Alpine(D))
    lb = bound_X.Alpine()[0]*np.ones(dimension.Alpine(D))
    optimizer = APSO(fitness=benchmark.Alpine,
                      D=dimension.Alpine(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Alpine()
    loss_curves[:, item] += optimizer.loss_curve
    

    item = item + 1
    ub = bound_X.Michalewicz()[1]*np.ones(dimension.Michalewicz())
    lb = bound_X.Michalewicz()[0]*np.ones(dimension.Michalewicz())
    optimizer = APSO(fitness=benchmark.Michalewicz,
                      D=dimension.Michalewicz(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Michalewicz(dimension.Michalewicz())
    loss_curves[:, item] += optimizer.loss_curve
    

    item = item + 1
    ub = bound_X.Exponential()[1]*np.ones(dimension.Exponential(D))
    lb = bound_X.Exponential()[0]*np.ones(dimension.Exponential(D))
    optimizer = APSO(fitness=benchmark.Exponential,
                      D=dimension.Exponential(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Exponential()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Schaffer()[1]*np.ones(dimension.Schaffer())
    lb = bound_X.Schaffer()[0]*np.ones(dimension.Schaffer())
    optimizer = APSO(fitness=benchmark.Schaffer,
                      D=dimension.Schaffer(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Schaffer()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.BentCigar()[1]*np.ones(dimension.BentCigar(D))
    lb = bound_X.BentCigar()[0]*np.ones(dimension.BentCigar(D))
    optimizer = APSO(fitness=benchmark.BentCigar,
                      D=dimension.BentCigar(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.BentCigar()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Bohachevsky1()[1]*np.ones(dimension.Bohachevsky1())
    lb = bound_X.Bohachevsky1()[0]*np.ones(dimension.Bohachevsky1())
    optimizer = APSO(fitness=benchmark.Bohachevsky1,
                      D=dimension.Bohachevsky1(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Bohachevsky1()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Elliptic()[1]*np.ones(dimension.Elliptic(D))
    lb = bound_X.Elliptic()[0]*np.ones(dimension.Elliptic(D))
    optimizer = APSO(fitness=benchmark.Elliptic,
                      D=dimension.Elliptic(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Elliptic()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.DropWave()[1]*np.ones(dimension.DropWave())
    lb = bound_X.DropWave()[0]*np.ones(dimension.DropWave())
    optimizer = APSO(fitness=benchmark.DropWave,
                      D=dimension.DropWave(), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.DropWave()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.CosineMixture()[1]*np.ones(dimension.CosineMixture(D))
    lb = bound_X.CosineMixture()[0]*np.ones(dimension.CosineMixture(D))
    optimizer = APSO(fitness=benchmark.CosineMixture,
                      D=dimension.CosineMixture(dimension.CosineMixture(D)), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.CosineMixture(dimension.CosineMixture(D))
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.Ellipsoidal(dimension.Ellipsoidal(D))[1]*np.ones(dimension.Ellipsoidal(D))
    lb = bound_X.Ellipsoidal(dimension.Ellipsoidal(D))[0]*np.ones(dimension.Ellipsoidal(D))
    optimizer = APSO(fitness=benchmark.Ellipsoidal,
                      D=dimension.Ellipsoidal(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.Ellipsoidal()
    loss_curves[:, item] += optimizer.loss_curve
    
    
    item = item + 1
    ub = bound_X.LevyandMontalvo1()[1]*np.ones(dimension.LevyandMontalvo1(D))
    lb = bound_X.LevyandMontalvo1()[0]*np.ones(dimension.LevyandMontalvo1(D))
    optimizer = APSO(fitness=benchmark.LevyandMontalvo1,
                      D=dimension.LevyandMontalvo1(D), P=P, G=G, ub=ub, lb=lb)
    st = time.time()
    optimizer.opt()
    ed = time.time()
    F_table[t, item] = optimizer.gbest_F
    table[item]['avg'] += optimizer.gbest_F
    table[item]['time'] += ed - st
    table[item]['ideal'] = ideal_F.LevyandMontalvo1()
    loss_curves[:, item] += optimizer.loss_curve

    
    print(t+1)

loss_curves = loss_curves / run_times
loss_curves = pd.DataFrame(loss_curves)
loss_curves.columns = ['Sphere', 'Rastrigin', 'Ackley', 'Griewank', 'Schwefel P2.22',
                       'Rosenbrock', 'Sehwwefel P2.21', 'Quartic', 'Schwefel P1.2', 'Penalized 1',
                       'Penalized 2', 'Schwefel P2.26', 'Step', 'Kowalik', 'Shekel Foxholes',
                       'Goldstein-Price', 'Shekel 5', 'Branin', 'Hartmann 3', 'Shekel 7',
                       'Shekel 10', 'Six-Hump Camel-Back', 'Hartmann 6', 'Zakharov', 'Sum Squares',
                       'Alpine', 'Michalewicz', 'Exponential', 'Schaffer', 'Bent Cigar',
                       'Bohachevsky 1', 'Elliptic', 'Drop Wave', 'Cosine Mixture', 'Ellipsoidal',
                       'Levy and Montalvo 1']
loss_curves.to_csv('loss_curves(APSO).csv')

table.loc[['avg', 'time']] = table.loc[['avg', 'time']] / run_times
table.loc['worst'] = F_table.max(axis=0)
table.loc['best'] = F_table.min(axis=0)
table.loc['std'] = F_table.std(axis=0)
table.columns = ['Sphere', 'Rastrigin', 'Ackley', 'Griewank', 'Schwefel P2.22',
                 'Rosenbrock', 'Sehwwefel P2.21', 'Quartic', 'Schwefel P1.2', 'Penalized 1',
                 'Penalized 2', 'Schwefel P2.26', 'Step', 'Kowalik', 'Shekel Foxholes',
                 'Goldstein-Price', 'Shekel 5', 'Branin', 'Hartmann 3', 'Shekel 7',
                 'Shekel 10', 'Six-Hump Camel-Back', 'Hartmann 6', 'Zakharov', 'Sum Squares',
                 'Alpine', 'Michalewicz', 'Exponential', 'Schaffer', 'Bent Cigar',
                 'Bohachevsky 1', 'Elliptic', 'Drop Wave', 'Cosine Mixture', 'Ellipsoidal',
                 'Levy and Montalvo 1']
table.to_csv('table(APSO).csv')