#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/26 22:36
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : AnD.py
# @Statement : A many-objective evolutionary algorithm with angle-based selection and shift-based density estimation
# @Reference : Liu Z Z, Wang Y, Huang P Q. AnD: A many-objective evolutionary algorithm with angle-based selection and shift-based density estimation[J]. Information Sciences, 2020, 509: 400-419.
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def cal_obj(pop, nobj):
    # 0 <= x <= 1
    g = 100 * (pop.shape[1] - nobj + 1 + np.sum((pop[:, nobj - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (pop[:, nobj - 1:] - 0.5)), axis=1))
    objs = np.zeros((pop.shape[0], nobj))
    temp_pop = pop[:, : nobj - 1]
    for i in range(nobj):
        f = 0.5 * (1 + g)
        f *= np.prod(temp_pop[:, : temp_pop.shape[1] - i], axis=1)
        if i > 0:
            f *= 1 - temp_pop[:, temp_pop.shape[1] - i]
        objs[:, i] = f
    return objs


def crossover(mating_pool, lb, ub, pc, eta_c):
    # simulated binary crossover (SBX)
    (noff, nvar) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, nvar))
    mu = np.random.random((nm, nvar))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, (nm, nvar))
    beta[np.random.random((nm, nvar)) < 0.5] = 1
    beta[np.tile(np.random.random((nm, 1)) > pc, (1, nvar))] = 1
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, pm, eta_m):
    # polynomial mutation
    (npop, nvar) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, nvar)) < pm / nvar
    mu = np.random.random((npop, nvar))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def calSD(objs, obj):
    # calculate the shift density
    npop = objs.shape[0]
    temp_objs = np.max((objs, np.tile(obj, (npop, 1))), axis=0)
    dis = np.full(npop, np.inf)
    for i in range(npop):
        if np.any(objs[i] != obj):
            dis[i] = np.sqrt(np.sum((obj - temp_objs[i]) ** 2))
    d = np.sort(dis)[int(np.sqrt(npop))]
    return 1 / (d + 2)


def environmental_selection(objs, npop):
    # AnD environmental selection
    zmin = np.min(objs, axis=0)  # ideal point
    zmax = np.max(objs, axis=0)  # nadir point
    objs = (objs - zmin) / (zmax - zmin)
    cosine = 1 - cdist(objs, objs, 'cosine')
    eye = np.arange(len(cosine))
    cosine[eye, eye] = 0
    angle = np.arccos(cosine)  # the angle between each pair of objectives
    remain = np.full(cosine.shape[0], True)

    while np.sum(remain) > npop:
        temp_angle = angle[remain][:, remain]
        temp_objs = objs[remain]
        flag = np.where(temp_angle == np.min(temp_angle))
        ind1 = flag[0][0]
        ind2 = flag[1][0]
        SD1 = calSD(temp_objs, temp_objs[ind1])
        SD2 = calSD(temp_objs, temp_objs[ind2])
        if SD1 < SD2:
            remain[np.where(remain)[0][ind2]] = False
        else:
            remain[np.where(remain)[0][ind1]] = False
    return remain


def dominates(obj1, obj2):
    # determine whether obj1 dominates obj2
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            return False
        elif obj1[i] != obj2[i]:
            sum_less += 1
    return sum_less > 0


def main(npop, iter, lb, ub, nobj=3, pc=1, pm=1, eta_c=20, eta_m=20):
    """
    The main function
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nobj: the dimension of objective space
    :param pc: crossover probability (default = 1)
    :param pm: mutation probability (default = 1)
    :param eta_c: spread factor distribution index (default = 20)
    :param eta_m: perturbance factor distribution index (default = 20)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = cal_obj(pop, nobj)  # objectives

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 50 == 0:
            print('Iteration: ' + str(t + 1) + ' completed.')

        # Step 2.1. Mating selection + crossover + mutation
        mating_pool = np.random.permutation(pop)
        off = crossover(mating_pool, lb, ub, pc, eta_c)
        off = mutation(off, lb, ub, pm, eta_m)
        off_objs = cal_obj(off, nobj)

        # Step 2.2. Environmental selection
        remain = environmental_selection(np.concatenate((objs, off_objs), axis=0), npop)  # the remaining individuals to the next generation
        pop = np.concatenate((pop, off), axis=0)[remain]
        objs = np.concatenate((objs, off_objs), axis=0)[remain]

    # Step 3. Sort the results
    dom = np.full(npop, False)
    for i in range(npop - 1):
        for j in range(i, npop):
            if not dom[i] and dominates(objs[j], objs[i]):
                dom[i] = True
            if not dom[j] and dominates(objs[i], objs[j]):
                dom[j] = True
    pf = objs[~dom]
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.view_init(45, 45)
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    z = [o[2] for o in pf]
    ax.scatter(x, y, z, color='red')
    ax.set_xlabel('objective 1')
    ax.set_ylabel('objective 2')
    ax.set_zlabel('objective 3')
    plt.title('The Pareto front of DTLZ1')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(100, 400, np.array([0] * 7), np.array([1] * 7))
