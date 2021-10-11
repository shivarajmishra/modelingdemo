import pandas as pd
import numpy as np

import utilities as util

def GetColumn(
		fileName, baseCol, startYear, yearRange, discount,
		compareCol=False, groupIndex=False):

	df = pd.read_csv(
		fileName + '_mm.csv',
		index_col=list(range(6)),
		header=list(range(1)))
	
	if compareCol != False:
		df[baseCol] = df[compareCol] - df[baseCol]
	df = df[[baseCol]]
	df = df.rename(columns={baseCol : 'value'})
	
	if discount:
		df['value'] = df['value'] * ((1.0 + discount) ** (startYear - df.index.get_level_values('year')))
	
	df = util.IndexToFront(df, 'year')
	df = df.loc[(slice(startYear + yearRange[0], startYear + yearRange[1])), :]
	
	if groupIndex:
		df = df.groupby(groupIndex).sum()
		
	return df['value']


def LoadRelevantTable(interventionList, groupIndex, metric, startYear, discount, yearRange):
	metricBAU = ('bau_' + metric)
	df = pd.DataFrame()
	df['bau'] = GetColumn(
		interventionList[0],
		metricBAU, startYear, yearRange, discount,
		compareCol=False, groupIndex=groupIndex)
	
	for intervention in interventionList:
		df[intervention] = GetColumn(
			intervention,
			metricBAU, startYear, yearRange, discount,
			compareCol=metric, groupIndex=groupIndex)
	return df


def AggregateTableByRanges(dfIn, aggIndex, aggRanges, groupIndex):
	df = pd.DataFrame()
	for ranges in aggRanges:
		dfRange = dfIn.copy()
		dfIndex = dfRange.index.to_frame()
		minAge = max(dfIndex[aggIndex].min(), ranges[0])
		maxAge = min(dfIndex[aggIndex].max(), ranges[1])
		print("aggIndex", aggIndex, ranges[0], ranges[1], groupIndex, minAge, maxAge)
		if minAge <= maxAge:
			dfRange = dfRange.loc[(slice(minAge, maxAge)), :]
			dfRange = dfRange.groupby(util.ListRemove(groupIndex, aggIndex)).sum()
			indexName = '{} to {}'.format(ranges[0], ranges[1])
			dfRange = util.AddIndexLevel(dfRange, aggIndex, [indexName])
			df = df.append(dfRange)
	df = df.sort_index()
	return df


def MakeAggregateTable(interventionList, aggIndex, aggRanges, yearFilterRanges, metric, startYear, discount=False):
	print("==== MakeAggregateTable ====", aggIndex, discount)
	# Config
	groupIndex = [aggIndex, 'sex', 'strata']
	
	# Stuff
	for yearRange in yearFilterRanges:
		for dis in [discount, False]:
			outName = 'process/out_{}_{}_year_{}-{}_discount_{}'.format(
				metric, aggIndex, yearRange[0], yearRange[1], dis)
			df = LoadRelevantTable(interventionList, groupIndex, metric, startYear, dis, yearRange)
			df = AggregateTableByRanges(df, aggIndex, aggRanges, groupIndex)
			
			util.OutputToFile(df, outName, head=False)


def DoProcess(interventionList, metrics, startYear=2020, ageRanges=[], yearRanges=[], discount={}):
	yearOfBirthRanges = [[startYear - x[1], startYear - x[0]] for x in ageRanges]
	for metric in metrics:
		MakeAggregateTable(
			interventionList, 'age', ageRanges, yearRanges,
			metric, startYear, discount.get(metric))
		MakeAggregateTable(
			interventionList, 'year_of_birth', yearOfBirthRanges, yearRanges,
			metric, startYear, discount.get(metric))


DoProcess(
	[
		'denicmedia',
		'denicmediaretailnosmoke',
		'denicotine',
		'retail',
		'smokefree',
	],
	[
		'HALY',
		'total_spent',
		'total_income',
		'deaths',
	],
	discount={
		'HALY' : 0.03,
		'total_spent' : 0.03,
		'total_income' : 0.03,
		'deaths' : 0.03,
	},
	startYear = 2020,
	ageRanges = [[0, 109], [0, 19], [20, 44], [45, 64], [65, 109]],
	yearRanges = [[0, 5], [6, 10], [11, 20], [21, 110], [0, 110]],
)
