#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import os

folder = "/Users/akulbansal/Library/CloudStorage/OneDrive-NorthwesternUniversity/NU_office/r/stoch/stoch_code/sddp_remote/sddp_data/data/"
prefix = "gep"


if __name__ == "__main__":

	HORIZON = 25	#change the horizone here if you want to generate data in some other way
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
	rate = 0.08     
	price_mean = np.array([9.37000, 9.79257, 10.23477, 10.69770, 11.18226, 11.68951, 12.22057, 12.77657, 13.35693, 13.96636, 14.21, 14.52, 14.83, 15.14, 15.45, 15.76, 16.07, 16.38, 16.69, 17.00, 17.31, 17.62, 17.93, 18.24, 18.55])
	price_std  = np.array([0.0, 0.9477973, 1.2146955, 1.4957925, 1.7848757, 2.0798978, 2.3814051, 2.6911019, 3.0063683, 3.3382216, 3.56, 3.81, 4.06, 4.31, 4.56, 4.81, 5.06, 5.31, 5.56, 5.81, 6.06, 6.31, 6.56, 6.81, 7.06])


	nType = len(GENERATOR_LIST)
	nUnit = sum(MAX_UNIT)
	D0 = 0.57

	
	#generate a pandas data frame for fixed cost
	cols = ["Time"] + list(GENERATOR_LIST.values()) + ["unmet"]
	df = pd.DataFrame(columns = cols)

	for row_index in range(HORIZON):
		time = row_index+1
		totalCost = np.multiply(CONSTRUCTION_COST, MAX_CAPACITY)
		finalCost = totalCost/ (1 + rate) ** (time-1)
		row = [time] + list(finalCost) + [PENALTY_COST / (1 + rate) ** (time-1)]
		df = df.append(pd.Series(row, index=df.columns), ignore_index=True)
	
	print(df)
	df.to_csv(folder + "gep_fixed_cost_jou.csv", index = False)

	#generate pandas data frame for operational costs
	cols = ["Time"] + list(GENERATOR_LIST.values())
	df2 = pd.DataFrame(columns = cols)

	for row_index in range(HORIZON):
		time = row_index+1
		totalCost = np.multiply(FUEL_PRICE, RATIO) * (1 + FUEL_PRICE_GROWTH) ** (time -1)
		totalCost += OPERATING_COST * (1 + OPER_COST_GROWTH) ** (time-1)
		finalCost = totalCost/ (1 + rate) ** (time-1)
		row = [time] + list(finalCost)
		df2 = df2.append(pd.Series(row, index=df2.columns), ignore_index=True)
	
	print(df2)
	df2.to_csv(folder + "gep_varCost_jou.csv", index = False)
	