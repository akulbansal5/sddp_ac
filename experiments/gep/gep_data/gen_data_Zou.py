#!/usr/bin/env python
import numpy as np
from scipy.stats import truncnorm
import os

if __name__ == "__main__":

    HORIZON = 5
    GENERATOR_LIST = {0: 'BaseLoad', 1: 'CC', 2: 'CT', 3: 'Nuclear', 4: 'Wind', 5: 'IGCC'}
    MAX_OUTPUT = np.array([1130.0, 390.0, 380.0, 1180.0, 175.0, 560.0])
    MAX_UNIT = np.array([4, 10, 10, 1, 45, 4], np.int32)
    SUBPERIOD = 3
    SUBPERIOD_HOUR = np.array([271.0, 6556.0, 1933.0])

    CONSTRUCTION_COST = np.array([1.446, 0.795, 0.575, 1.613, 1.650, 1.671])
    MAX_CAPACITY = np.array([1200.0, 400.0, 400.0, 1200.0, 500.0, 600.0])

    FUEL_PRICE = np.array([3.37, 9.37, 9.37, 0.93e-3, 0, 3.37]) * 1e-6
    RATIO = [8.844 / 0.4, 7.196 / 0.56, 10.842 / 0.4, 10.400 / 0.45, 0.0, 8.613 / 0.48]
    FUEL_PRICE_GROWTH = 0.02
    OPERATING_COST = np.array([4.7, 2.11, 3.66, 0.51, 5.00, 2.98]) * 1e-6
    OPER_COST_GROWTH = 0.03
    PENALTY_COST = 1e-1
    LAMBDA = np.array([1.38, 1.04, 0.80])
    HOURS_PER_YEAR = 8760.0
    rate = 0.08     # interest rate
    #price_mean = np.ones(HORIZON) * 9.37
    price_mean = np.array([9.37000, 9.79257, 10.23477, 10.69770, 11.18226, 11.68951, 12.22057, 12.77657, 13.35693, 13.96636])
    price_std = np.array([0.0, 0.9477973, 1.2146955, 1.4957925, 1.7848757, 2.0798978, 2.3814051, 2.6911019, 3.0063683, 3.3382216])
    #price_std = np.zeros(HORIZON)


    nType = len(GENERATOR_LIST)     # number of technology
    nUnit = sum(MAX_UNIT)           # sum of maximum construction
    binLength = np.floor(np.log2(MAX_UNIT)).astype(int)
    sumLength = sum(binLength) + nType

    numFWsample = 2
    scenPerStage = 100

    numScen = np.ones(HORIZON - 1, np.int32) * scenPerStage
    numScen = np.insert(numScen, 0, 1)

    initState = np.zeros(sumLength)
    valueLB = np.zeros(HORIZON)

    D0 = 0.57
    scenarios = [[0] * numScen[i] for i in xrange(HORIZON)]
    scenarios[0][0] = [0.0] * 2 * SUBPERIOD  # (d, -d)

    demand_tree = [[0] * numScen[i] for i in xrange(HORIZON)]
    demand_tree[0][0] = [0.0] * SUBPERIOD

    for t in xrange(1, HORIZON):
        sample = D0 * (1.015 ** t - 1.005 ** t) * np.random.sample((scenPerStage,)) + D0 * 1.005 ** t
        scenarios[t] = list(sample)
        for j in xrange(numScen[t]):
            temp = LAMBDA * max(scenarios[t][j] - D0, 0.0) * 1e9 / HOURS_PER_YEAR
            demand_tree[t][j] = list(temp)
            temp = np.concatenate((temp, -temp))
            scenarios[t][j] = list(temp)

	price = [[0] * numScen[t] for t in xrange(HORIZON)]
	price[0][0] = price_mean[0] * 1e-12 / 1.028
	
	for t in xrange(1, HORIZON):
		sample = (price_mean[t] + price_std[t] * truncnorm.rvs(-1, 1, size=numScen[t])) * 1e-12 / 1.028
		price[t] = list(sample)
		
    ''' Parameters used by tree model '''

    treeDataDir = "test_tree_model/data_" + str(HORIZON) + "/"
    if not os.path.exists(treeDataDir):
        os.makedirs(treeDataDir)

    # number of stages
    myFile = open(treeDataDir + "numStage.dat", "w")
    myFile.write(str(HORIZON))
    myFile.close()

    # number of generators
    myFile = open(treeDataDir + "numGen.dat", "w")
    myFile.write(str(nType))
    myFile.close()

    # number of subperiods
    myFile = open(treeDataDir + "numSub.dat", "w")
    myFile.write(str(SUBPERIOD))
    myFile.close()

    # number of children
    myFile = open(treeDataDir + "numChi.dat", "w")
    myFile.write(str(scenPerStage))
    myFile.close()

    # number of nodes
    numNode = (scenPerStage ** HORIZON - 1) / (scenPerStage - 1)
    myFile = open(treeDataDir + "numNode.dat", "w")
    myFile.write(str(numNode))
    myFile.close()

    # max output
    myFile = open(treeDataDir + "maxOutput.dat", "w")
    myFile.write("[")
    for i in xrange(MAX_OUTPUT.shape[0]):
        myFile.write(str(MAX_OUTPUT[i]))
        if i != MAX_OUTPUT.shape[0] - 1:
            myFile.write(",")
    myFile.write("]")
    myFile.close()

    # max units
    myFile = open(treeDataDir + "maxUnit.dat", "w")
    myFile.write("[")
    for i in xrange(MAX_UNIT.shape[0]):
        myFile.write(str(MAX_UNIT[i]))
        if i != MAX_UNIT.shape[0] - 1:
            myFile.write(",")
    myFile.write("]")
    myFile.close()

    # scenarios
    myFile = open(treeDataDir + "demand.dat", "w")
    myFile.write(str(scenarios))
    myFile.close()

    # xCoef of tree model
    prob = 1.0 / scenPerStage
    xCoef = np.zeros((HORIZON, nType))
    totalCost = np.multiply(CONSTRUCTION_COST, MAX_CAPACITY)
    for t in np.arange(HORIZON):
        xCoef[t] = (prob ** t) * totalCost / (1 + rate) ** t

    myFile = open(treeDataDir + "xCoef.dat", "w")
    myFile.write("[")
    for row in xrange(xCoef.shape[0]):
        myFile.write("[")
        for column in xrange(xCoef.shape[1]):
            myFile.write(str(xCoef[row, column]))
            if column != xCoef.shape[1] - 1:
                myFile.write(",")
        myFile.write("]")
        if row != xCoef.shape[0] - 1:
            myFile.write(",")
    myFile.write("]")
    myFile.close()

    # yCoef of tree model
    yCoef = []
    for t in np.arange(HORIZON):
    	yCoef_t = [ [0] * SUBPERIOD for n in xrange(numScen[t]) ]
    	for n in xrange(numScen[t]):
    		for k in xrange(SUBPERIOD):
    			temp = np.multiply(FUEL_PRICE, RATIO) * (1 + FUEL_PRICE_GROWTH) ** t
    			for i in [1, 2]:
    				temp[i] = price[t][n] * RATIO[i]
    			temp = temp + OPERATING_COST * (1 + OPER_COST_GROWTH) ** t
    			temp = (prob ** t) * SUBPERIOD_HOUR[k] * temp / (1 + rate) ** t
    			yCoef_t[n][k] = list(temp)
    	yCoef.append(yCoef_t)
    #print yCoef
            
    myFile = open(treeDataDir + "yCoef.dat", "w")
    myFile.write(str(yCoef))
    myFile.close()

    # zCoef of tree model
    zCoef = np.zeros((HORIZON, SUBPERIOD))
    for t in np.arange(HORIZON):
        zCoef[t] = (prob ** t) * SUBPERIOD_HOUR * PENALTY_COST / (1 + rate) ** t
    
    myFile = open(treeDataDir + "zCoef.dat", "w")
    myFile.write("[")
    for t in xrange(zCoef.shape[0]):
        myFile.write("[")
        for k in xrange(zCoef.shape[1]):
            myFile.write(str(zCoef[t, k]))
            if k != zCoef.shape[1] - 1:
                myFile.write(",")
        myFile.write("]")
        if t != zCoef.shape[0] - 1:
            myFile.write(",")
    myFile.write("]")
    myFile.close()


    ''' Parameters used by SDDP '''

    dataDir = "bin/test_data/" + str(HORIZON) + "_" + str(scenPerStage) + "/"
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)

    # number of stages
    myFile = open(dataDir + "numStage.dat", "w")
    myFile.write(str(HORIZON))
    myFile.close()

    # initial state variable
    myFile = open(dataDir + "initState.dat", "w")
    myFile.write("[")
    for i in xrange(initState.shape[0]):
        myFile.write(str(initState[i]))
        if i != initState.shape[0] - 1:
            myFile.write(",")
    myFile.write("]")
    myFile.close()

    # number of forward samples
    myFile = open(dataDir + "numFWsample.dat", "w")
    myFile.write(str(numFWsample))
    myFile.close()

    # number of scenarios at each stage
    myFile = open(dataDir + "numScen.dat", "w")
    myFile.write("[")
    for i in xrange(numScen.shape[0]):
        myFile.write(str(numScen[i]))
        if i != numScen.shape[0] - 1:
            myFile.write(",")
    myFile.write("]")
    myFile.close()

    # lower bounds for value functions
    myFile = open(dataDir + "valueLB.dat", "w")
    myFile.write("[")
    for i in xrange(valueLB.shape[0]):
        myFile.write(str(valueLB[i]))
        if i != valueLB.shape[0] - 1:
            myFile.write(",")
    myFile.write("]")
    myFile.close()

    # scenarios
    myFile = open(dataDir + "scenarios.dat", "w")
    myFile.write(str(scenarios))
    myFile.close()

    # y2Coef scenarios
    y2Scenarios = [ [0] * numScen[t] for t in xrange(HORIZON) ]
    for t in xrange(HORIZON):
    	for n in xrange(numScen[t]):
    		temp = [];
    		for k in xrange(SUBPERIOD):
    			temp = temp + list(np.array(yCoef[t][n][k]) * (np.reciprocal(prob) ** t))
    			temp.append(zCoef[t][k] * (prob ** (-t)))
    		y2Scenarios[t][n] = temp

    myFile = open(dataDir + "y2Scenarios.dat", "w")
    myFile.write(str(y2Scenarios))
    myFile.close()

    #print y2Scenarios

    ''' Construct data for objective function c x + b1 y1 + b2 y2 '''
    # construct c (coefficients for x state variabls)
    # doesn't show up in objective funcion, thus all zeros
    xCoef = np.zeros(sumLength)
    print "x.shape", xCoef.shape

    myFile = open(dataDir + "xCoef.dat", "w")
    myFile.write("[")
    for t in xrange(HORIZON):
        myFile.write("[")
        for row in xrange(xCoef.shape[0]):
            myFile.write(str(xCoef[row]))
            if row != xCoef.shape[0] - 1:
                myFile.write(",")
        myFile.write("]")
        if t != HORIZON - 1:
            myFile.write(",")
    myFile.write("]")
    myFile.close()

    # construct b1 (coefficients for y1 integer variabls)
    y1Coef = np.zeros((HORIZON, nType))
    totalCost = np.multiply(CONSTRUCTION_COST, MAX_CAPACITY)
    for t in np.arange(HORIZON):
        y1Coef[t] = totalCost / (1 + rate) ** t

    print "y1.shape", y1Coef.shape

    myFile = open(dataDir + "y1Coef.dat", "w")
    myFile.write("[")
    for row in xrange(y1Coef.shape[0]):
        myFile.write("[")
        for column in xrange(y1Coef.shape[1]):
            myFile.write(str(y1Coef[row, column]))
            if column != y1Coef.shape[1] - 1:
                myFile.write(",")
        myFile.write("]")
        if row != y1Coef.shape[0] - 1:
            myFile.write(",")
    myFile.write("]")
    myFile.close()

    # construct b2 (coefficients for y2 continuous variables
    # by default, we choose the first scenario to construct the problem
    y2Coef = np.zeros((HORIZON, SUBPERIOD * (nType + 1)))
    for t in np.arange(HORIZON):
        y2Coef[t] = y2Scenarios[t][0]

    print "y2.shape", y2Coef.shape

    myFile = open(dataDir + "y2Coef.dat", "w")
    myFile.write("[")
    for row in xrange(y2Coef.shape[0]):
        myFile.write("[")
        for column in xrange(y2Coef.shape[1]):
            myFile.write(str(y2Coef[row, column]))
            if column != y2Coef.shape[1] - 1:
                myFile.write(",")
        myFile.write("]")
        if row != y2Coef.shape[0] - 1:
            myFile.write(",")
    myFile.write("]")
    myFile.close()

    ''' Construct data for constraints A x + W1 y1 + W2 y2 + B z >= rhs '''
    P = np.zeros((nType, sumLength))
    for i in xrange(nType):
        P[i, sum(binLength[0:i]) + i: sum(binLength[0:i + 1]) + i + 1] = 2**np.arange(binLength[i] + 1)

    # construct matrix A (x variable)
    A = np.vstack((np.zeros((4 * nType, sumLength)),
                   P, -P,
                   np.zeros((2 * SUBPERIOD, sumLength))))

    print "A.shape:", A.shape

    myFile = open(dataDir + "Amatrix.dat", "w")
    myFile.write("[")
    for t in np.arange(HORIZON):
        myFile.write("[")
        for row in xrange(A.shape[0]):
            myFile.write("[")
            for column in xrange(A.shape[1]):
                myFile.write(str(A[row, column]))
                if column != A.shape[1] - 1:
                    myFile.write(",")
            myFile.write("]")
            if row != A.shape[0] - 1:
                myFile.write(",")
        myFile.write("]")
        if t != HORIZON - 1:
            myFile.write(",")
    myFile.write("]")
    myFile.close()

    # construct matrix W1 (y1 integer variables)
    I = np.identity(nType)
    W1 = np.vstack((I, I, I, -I, -I, I,
                    np.zeros((2 * SUBPERIOD, nType))))

    print "W1.shape:", W1.shape

    myFile = open(dataDir + "W1matrix.dat", "w")
    myFile.write("[")
    for t in np.arange(HORIZON):
        myFile.write("[")
        for row in xrange(W1.shape[0]):
            myFile.write("[")
            for column in xrange(W1.shape[1]):
                myFile.write(str(W1[row, column]))
                if column != W1.shape[1] - 1:
                    myFile.write(",")
            myFile.write("]")
            if row != W1.shape[0] - 1:
                myFile.write(",")
        myFile.write("]")
        if t != HORIZON - 1:
            myFile.write(",")
    myFile.write("]")
    
    # construct matrix W2 (y2 continuous variable)
    temp = -1.0 / MAX_OUTPUT
    R = np.diagflat([temp, temp, temp])
    for k in xrange(SUBPERIOD):
        R = np.insert(R, nType * (k + 1), 0.0, axis=1)

    row1 = np.hstack((np.ones(nType + 1),
                      np.zeros((SUBPERIOD - 1) * (nType + 1))))
    row2 = np.hstack((np.zeros(nType + 1),
                      np.ones(nType + 1),
                      np.zeros(nType + 1)))
    row3 = np.hstack((np.zeros((SUBPERIOD - 1) * (nType + 1)),
                      np.ones(nType + 1)))
    demandLHS = np.vstack((row1, row2, row3, -row1, -row2, -row3))

    W2 = np.vstack((R, np.zeros((3 * nType, SUBPERIOD * (nType + 1))), demandLHS))

    print "W2.shape:", W2.shape

    myFile = open(dataDir + "W2matrix.dat", "w")
    myFile.write("[")
    for t in np.arange(HORIZON):
        myFile.write("[")
        for row in xrange(W2.shape[0]):
            myFile.write("[")
            for column in xrange(W2.shape[1]):
                myFile.write(str(W2[row, column]))
                if column != W2.shape[1] - 1:
                    myFile.write(",")
            myFile.write("]")
            if row != W2.shape[0] - 1:
                myFile.write(",")
        myFile.write("]")
        if t != HORIZON - 1:
            myFile.write(",")
    myFile.write("]")
    myFile.close()

    # construct B matrix (z variables, local copy of state variable from last period)
    B = np.vstack((P, P, P, -P, -P, P,
                   np.zeros((2 * SUBPERIOD, sumLength))))

    print "B.shape:", B.shape

    myFile = open(dataDir + "Bmatrix.dat", "w")
    myFile.write("[")
    for t in np.arange(HORIZON):
        myFile.write("[")
        for row in xrange(B.shape[0]):
            myFile.write("[")
            for column in xrange(B.shape[1]):
                myFile.write(str(B[row, column]))
                if column != B.shape[1] - 1:
                    myFile.write(",")
            myFile.write("]")
            if row != B.shape[0] - 1:
                myFile.write(",")
        myFile.write("]")
        if t != HORIZON - 1:
            myFile.write(",")
    myFile.write("]")
    myFile.close()

    # construct rhs
    rhs = np.hstack((np.zeros((SUBPERIOD * nType)), -MAX_UNIT,
                     np.zeros(2 * nType + 2 * SUBPERIOD)))

    print "rhs.shape:", rhs.shape

    myFile = open(dataDir + "bRhs.dat", "w")
    myFile.write("[")
    for t in np.arange(HORIZON):
        myFile.write("[")
        for row in xrange(rhs.shape[0]):
            myFile.write(str(rhs[row]))
            if row != rhs.shape[0] - 1:
                myFile.write(",")
        myFile.write("]")
        if t != HORIZON - 1:
            myFile.write(",")
    myFile.write("]")
    myFile.close()

    # uncertain indices for the rhs
    uncertainIndexRHS = np.arange(rhs.shape[0] - 6, rhs.shape[0])

    myFile = open(dataDir + "uncertainIndexRHS.dat", "w")
    myFile.write("[")
    for i in xrange(uncertainIndexRHS.shape[0]):
        myFile.write(str(uncertainIndexRHS[i]))
        if i != uncertainIndexRHS.shape[0] - 1:
            myFile.write(",")
    myFile.write("]")
    myFile.close()