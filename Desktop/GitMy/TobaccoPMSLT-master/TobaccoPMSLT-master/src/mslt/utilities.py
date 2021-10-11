import pandas as pd
import numpy as np
import os

from pathlib import Path

fileCreated = {}
HEAD_MODE = False

def get_data_dir(population):
	here = Path(__file__).resolve()
	return here.parent / 'artifacts' / population


def UnstackDraw(df):
	df = df.sort_values(['year_start', 'age_start', 'sex', 'strata', 'draw'])
	df = df.set_index(['year_start',  'year_end', 'age_start', 'age_end', 'sex', 'strata', 'draw'])
	df = df.unstack(level='draw')
	df = df.droplevel(0, axis=1)
	col_frame = df.columns.to_frame()
	col_frame = col_frame.reset_index(drop=True)
	col_frame['draw'] = 'draw_' + col_frame['draw'].astype(str)
	df.columns = pd.Index(col_frame['draw'])
	df.columns.name = None
	df = df.reset_index()
	return df


def CrossDf(df1, df2):
	return (df1
		.assign(_cross_merge_key=1)
		.merge(df2.assign(_cross_merge_key=1), on="_cross_merge_key")
		.drop("_cross_merge_key", axis=1)
	)


def MakePath(path):
	if '/' not in path:
		return
	out_folder = os.path.dirname(path)
	if not os.path.exists(out_folder):
		MakePath(out_folder)
		os.mkdir(out_folder)


def OutputToFile(df, path, index=True):
	# Write dataframe to a file.
	# Appends dataframe when called with the same name.
	fullFilePath = path + '.csv'
	MakePath(path)
	
	if fileCreated.get(fullFilePath):
		# Append
		df.to_csv(fullFilePath, mode='a', header=False, index=index)
	else:
		fileCreated[fullFilePath] = True
		df.to_csv(fullFilePath, index=index) 
		if HEAD_MODE:
			df.head(100).to_csv(path + '_head' + '.csv', index=index) 


def SetStandardIndex(df):
	# Sorts age sex strata.
	df = df.reset_index()
	df = df.set_index(['age', 'sex', 'strata']).sort_index()
	return df


def AgeToCohorts(df, period, offset):
	indexNames = [x for x in df.index.names if x != None]
	df = df.reset_index()
	df = df[df['age'] % period == offset]
	df = df.set_index(indexNames)
	return df


def AddAge(df, max_age):
	indexNames = [x for x in df.index.names if x != None]
	if 'age' in indexNames:
		return df
	if len(indexNames) == 0:
		df = df.reset_index(drop=True)
	else:
		df = df.reset_index()
	
	if 'age' not in df.columns:
		df = CrossDf(df, pd.DataFrame({'age' : list(range(max_age))}))
	df = df.set_index(['age'] + indexNames)
	return df


def AddSex(df):
	indexNames = [x for x in df.index.names if x != None]
	if 'sex' in indexNames:
		return df
	if len(indexNames) == 0:
		df = df.reset_index(drop=True)
	else:
		df = df.reset_index()
	
	if 'sex' not in df.columns:
		df = CrossDf(df, pd.DataFrame({'sex' : ['female', 'male']}))
	df = df.set_index(['sex'] + indexNames)
	return df


df_strata = False
def SetStrataDf(df):
	global df_strata
	df_strata = df # global


def GetStrata():
	return list(df_strata['strata'])


def SetStrataFile(path):
	SetStrataDf(pd.read_csv(path))


def AddToIndex(df, indexName, indexVal):
	indexNames = [x for x in df.index.names if x != None]
	if indexName in indexNames:
		return df
	if len(indexNames) == 0:
		df = df.reset_index(drop=True)
	else:
		df = df.reset_index()

	if indexName not in df.columns:
		df = CrossDf(df, pd.DataFrame({indexName : [indexVal]}))
	df = df.set_index([indexName] + indexNames)
	return df


def AddStrata(df):
	indexNames = [x for x in df.index.names if x != None]
	if 'strata' in indexNames:
		return df
	if len(indexNames) == 0:
		df = df.reset_index(drop=True)
	else:
		df = df.reset_index()

	if 'strata' not in df.columns:
		if type(df_strata) == 'bool': # global
			SetStrataDf(pd.DataFrame({'strata' : ['all']}))
		df = CrossDf(df, df_strata)
	df = df.set_index(['strata'] + indexNames)
	return df


def ExpandAgeCategory(df, max_age):
	# Index must contain agecategory
	indexNames = df.index.names
	indexNames = [x for x in indexNames if (x != 'agecategory' and x != None)]
	df = df.reset_index()

	# Add age and filter out ages that are below the agecategory.
	df = CrossDf(df, pd.DataFrame({'age' : list(range(max_age))}))
	df = df[df['age'] >= df['agecategory']]

	# Add a copy of agecategory to the index so it can be grabbed with loc
	df['as_index'] = df['agecategory']
	df = df.set_index(['age', 'as_index'] + indexNames)
	df = df.loc[df.groupby(['age'] + indexNames)['agecategory'].idxmax()]
	df = df.drop(columns=['agecategory']).droplevel('as_index')
	return df


def ExpandToValueAtAge(df, special_age, max_age):
	indexNames = [x for x in df.index.names if x != None]
	if len(indexNames) == 0:
		df = df.reset_index(drop=True)
	else:
		df = df.reset_index()
	newStrata = [0, special_age, special_age + 1]
	if special_age == 0:
		newStrata = [0, 1]
	df = CrossDf(df, pd.DataFrame({'agecategory' : newStrata}))
	df = df.set_index(['agecategory'] + indexNames)
	for val in newStrata:
		if val != special_age:
			df.loc[val, :] = 0

	df = ExpandAgeCategory(df, max_age)
	return df


def ValueToValueRange(df, name, offset):
	indexNames = [x for x in df.index.names if (x != None and x != name)]
	if len(indexNames) == 0:
		df = df.reset_index(drop=True)
	else:
		df = df.reset_index()
	df['{}_end'.format(name)] = df[name] + offset
	df = df.rename(columns={name : '{}_start'.format(name)})
	df = df.set_index(['{}_start'.format(name), '{}_end'.format(name)] + indexNames)
	return df
