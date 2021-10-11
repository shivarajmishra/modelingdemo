"""Build compartment circut data tables."""

import pandas as pd
import numpy as np
import pathlib

from .uncertainty import sample_fixed_rate_from
from mslt.utilities import UnstackDraw, CrossDf, OutputToFile, AgeToCohorts, SetStandardIndex
from mslt.utilities import ExpandAgeCategory, ExpandToValueAtAge, AddAge, AddSex, AddStrata, SetStrataFile

import mslt.utilities as util

class Circuit:

	def __init__(self, art_nm, num_draws, data_dir, write_table, year_start, year_end, max_age, writeDiseases, prng):
		self.data_dir = '{}/circuit/'.format(data_dir)
		self.data_post_dir = '{}/circuit_post/'.format(data_dir)
		self.art_nm = art_nm
		self.write_table = write_table
		self.year_start = int(year_start)
		self.year_end = int(year_end + 1)
		self.max_age = max_age
		self.num_draws = num_draws

		self.cohortSize = 1
		self.cohortOffset = 0
		self.writeDiseases = writeDiseases

		self.tunnel_length = 20 # Todo, configurable.

		self.PostProcessRawInput()
		self.ProcessStandardInput()

	################ Standard input processing ################
	# This is generated from the raw input, and always has the same structure.

	def ProcessStandardInput(self):
		flow_suffix = '.csv'
		p_flow = pathlib.Path(self.data_post_dir + 'flow/')
		flow_paths = sorted(p_flow.glob('*{}'.format(flow_suffix)))
		modelArcFile = open("modelArcs.txt", "w")
		for flow_path in flow_paths:
			state_name = str(flow_path.name)[:-len(flow_suffix)]
			df = pd.read_csv(flow_path)
			if 'year_start' not in df.columns:
				df['year_start'] = self.year_start
				df['year_end'] = self.year_end
			else:
				df['year_start'] += self.year_start
				df['year_end'] += self.year_start

			df = df.rename(columns={'age': 'age_start'})
			df.insert(df.columns.get_loc('age_start') + 1, 'age_end', df['age_start'] + 1)

			df = df.set_index(['age_start', 'age_end', 'sex', 'strata', 'year_start', 'year_end'])
			modelArcFile.write('            {}: {}\n'.format(state_name, [x for x in df.columns.values if '_apc' not in x]))
			self.WriteByStatesWithApc(df, 'circuit.flow.{}_'.format(state_name))
		modelArcFile.close()

		if self.writeDiseases:
			rr_suffix = '_rr.csv'
			p_rr = pathlib.Path(self.data_post_dir + 'rr_disease/')
			rr_paths = sorted(p_rr.glob('*{}'.format(rr_suffix)))
			for rr_path in rr_paths:
				disease = str(rr_path.name)[:-len(rr_suffix)]
				df = pd.read_csv(rr_path)
				df['year_start'] = self.year_start
				df['year_end'] = self.year_end

				df = df.rename(columns={'age': 'age_start'})
				df.insert(df.columns.get_loc('age_start') + 1, 'age_end', df['age_start'] + 1)

				df = df.set_index(['age_start', 'age_end', 'sex', 'strata', 'year_start', 'year_end'])
				self.WriteByStatesWithApc(df, 'circuit.rr.{}_'.format(disease))

		p_prevalence = pathlib.Path(self.data_post_dir + 'prevalence/')
		prevalence_suffix = '.csv'
		prevalence_paths = sorted(p_prevalence.glob('*{}'.format(prevalence_suffix)))
		for prev_path in prevalence_paths:
			prevName = str(prev_path.name)[:-len(prevalence_suffix)]
			df_prevalence = pd.read_csv(prev_path)
			df_prevalence['year'] = self.year_start
			df_prevalence = df_prevalence.set_index(['age', 'sex', 'strata', 'year'])
			self.WriteByStatesWithApc(df_prevalence, 'circuit.{}.'.format(prevName))


	def ProcessApc(self, df):
		for col in df.columns:
			apcName = col + '_apc'
			if apcName in df.columns:
				df_year = pd.DataFrame({'year' : list(range(0, self.year_end - self.year_start))})
				df_col = CrossDf(df[[col, apcName]].reset_index(), df_year)
				df_col['year_start'] = df_col['year_start'] + df_col['year']
				df_col['year_end'] = df_col['year_start'] + 1
				df[col] = df_col[col] * (1 + df_col[apcName]) ** df_col['year']
		return df


	def WriteByStatesWithApc(self, df_states, path):
		for col in df_states.columns:
			if '_apc' not in col:
				df = df_states[[col]]
				apcName = col + '_apc'
				if apcName in df_states.columns:
					df_year = pd.DataFrame({'year' : list(range(0, self.year_end - self.year_start))})
					df_apc = CrossDf(df_states[[col, apcName]].reset_index(), df_year)
					df_apc['year_start'] = df_apc['year_start'] + df_apc['year']
					df_apc['year_end'] = df_apc['year_start'] + 1
					df_apc = df_apc.set_index(['age_start', 'age_end', 'sex', 'strata', 'year_start', 'year_end'])
					df = df_apc[[col]].mul((1 + df_apc[apcName]) ** df_apc['year'], axis=0)
				df = df.rename(columns={col : 'draw_0'})
				for x in range(self.num_draws):
					df['draw_{}'.format(x + 1)] = df['draw_0']
				df = df.reset_index()
				self.write_table(self.art_nm, path + col, df) 
	
	################ Raw input processing ################
	# The structure of this input changes depending on the model.

	def PostProcessRawInput(self):
		self.ProcessRawFlow()

		p_prevalence = pathlib.Path(self.data_dir + 'prevalence/')
		prevalence_suffix = '.csv'
		prevalence_paths = sorted(p_prevalence.glob('*{}'.format(prevalence_suffix)))
		for prev_path in prevalence_paths:
			prev_path = str(prev_path.name)[:-len(prevalence_suffix)]
			print('pevalence', prev_path)
			self.ProcessPrevalence(prev_path)
		
		if self.writeDiseases:
			self.ProcessRelativeRisk()


	def ProcessRawFlow(self):
		path = self.data_dir + 'flow/'
		path_out = self.data_post_dir + 'flow/'

		df_cs = pd.read_csv(path + 'cs.csv').set_index('agecategory')
		df_cscv = pd.read_csv(path + 'cscv.csv').set_index('agecategory')
		df_nscv = pd.read_csv(path + 'nscv.csv').set_index('agecategory')
		df_fscv = pd.read_csv(path + 'fscv.csv').set_index('agecategory')
		df_uptake = pd.read_csv(path + 'uptake.csv').set_index(['sex', 'strata'])
		df_apc = pd.read_csv(path + 'uptake_apc.csv').set_index(['sex', 'strata'])
		df_quit_exit = pd.read_csv(path + 'quit_smoking_relapse.csv').set_index(['agecategory', 'sex', 'strata'])

		df_cs = SetStandardIndex(AddSex(AddStrata(ExpandAgeCategory(df_cs, self.max_age))))
		df_cscv = SetStandardIndex(AddSex(AddStrata(ExpandAgeCategory(df_cscv, self.max_age))))
		df_nscv = SetStandardIndex(AddSex(AddStrata(ExpandAgeCategory(df_nscv, self.max_age))))
		df_fscv = SetStandardIndex(AddSex(AddStrata(ExpandAgeCategory(df_fscv, self.max_age))))
		df_uptake = SetStandardIndex(AddStrata(ExpandToValueAtAge(df_uptake, 20, self.max_age)))
		df_apc = SetStandardIndex(AddStrata(AddSex(ExpandToValueAtAge(df_apc, 20, self.max_age))))
		df_quit_exit = SetStandardIndex(AddSex(AddStrata(ExpandAgeCategory(df_quit_exit, self.max_age))))

		df_one = pd.DataFrame({'value' : [1]})
		df_one = SetStandardIndex(AddStrata(AddSex(AddAge(df_one, self.max_age))))

		OutputToFile(pd.concat([
			df_uptake.rename(columns={
				'UptakeProp' : 'cs', 'UptakePropBoth' : 'cscv', 'UptakeVaping' : 'nscv'}),
			df_apc.rename(columns={
				'UptakePropAPC' : 'cs_apc', 'UptakePropBothAPC' : 'cscv_apc', 'UptakeVapingAPC' : 'nscv_apc'})
		], axis=1), path_out + 'ns')

		OutputToFile(df_cs, path_out + 'cs')
		OutputToFile(df_cscv, path_out + 'cscv')
		OutputToFile(df_nscv, path_out + 'nscv')
		OutputToFile(df_fscv, path_out + 'fscv')

		for i in range(1, self.tunnel_length):
			OutputToFile(df_one.rename(columns={
				'value' : 'qsqv_{}'.format(i + 1) if i < (self.tunnel_length - 1) else 'fsfv'
			}), path_out + 'qsqv_{}'.format(i))
			OutputToFile(df_one.rename(columns={
				'value' : 'nsqv_{}'.format(i + 1) if i < (self.tunnel_length - 1) else 'nsfv'
			}), path_out + 'nsqv_{}'.format(i))

			df_qscv = df_quit_exit.copy()
			transName = 'qscv_{}'.format(i + 1) if i < (self.tunnel_length - 1) else 'fscv'
			if i < (self.tunnel_length - 1):
				df_qscv[transName] = 1 - df_qscv['QuitVapeProp'] - df_qscv['QuitSmokingRelapse']
				df_qscv = df_qscv.rename(columns={
					'QuitSmokingRelapse' : 'cs',
					'QuitVapeProp' : 'qsqv_{}'.format(i + 1)
				})
			else:
				df_qscv[transName] = 1 - df_qscv['QuitSmokingRelapse']
				df_qscv = df_qscv.drop(columns=['QuitVapeProp']).rename(columns={
					'QuitSmokingRelapse' : 'cs',
				})
			OutputToFile(df_qscv, path_out + 'qscv_{}'.format(i))
		

	def ProcessPrevalence(self, fileName):
		path = self.data_dir + 'prevalence/'
		path_out = self.data_post_dir + 'prevalence/'

		df_prev = pd.read_csv(path + '{}.csv'.format(fileName)).set_index(['agecategory', 'strata', 'sex'])
		df_prev = SetStandardIndex(ExpandAgeCategory(df_prev, self.max_age))
		df_prev = df_prev.reset_index()

		for i in range(1, self.tunnel_length):
			df_prev['qsqv_{}'.format(i)] = df_prev['totalFSFV'] * np.where(
				df_prev['age'] >= i + (self.tunnel_length - 1), 1 / (df_prev['age'] - (self.tunnel_length - 1)), 0)
			df_prev['qscv_{}'.format(i)] = df_prev['totalFSCV'] * np.where(
				df_prev['age'] >= i + (self.tunnel_length - 1), 1 / (df_prev['age'] - (self.tunnel_length - 1)), 0)
			df_prev['nsqv_{}'.format(i)] = df_prev['totalNSFV'] * np.where(
				df_prev['age'] >= i + (self.tunnel_length - 1), 1 / (df_prev['age'] - (self.tunnel_length - 1)), 0)
		
		df_prev['fsfv'] = df_prev['totalFSFV'] * np.where(
			df_prev['age'] >= 2*(self.tunnel_length - 1) + 1, (df_prev['age'] - 2*(self.tunnel_length - 1)) / (df_prev['age'] - (self.tunnel_length - 1)), 0)
		df_prev['fscv'] = df_prev['totalFSCV'] * np.where(
			df_prev['age'] >= 2*(self.tunnel_length - 1) + 1, (df_prev['age'] - 2*(self.tunnel_length - 1)) / (df_prev['age'] - (self.tunnel_length - 1)), 0)
		df_prev['nsfv'] = df_prev['totalNSFV'] * np.where(
			df_prev['age'] >= 2*(self.tunnel_length - 1) + 1, (df_prev['age'] - 2*(self.tunnel_length - 1)) / (df_prev['age'] - (self.tunnel_length - 1)), 0)
		
		df_prev = df_prev.set_index(['age', 'sex', 'strata'])

		df_prev = df_prev.drop(columns=['totalFSFV', 'totalFSCV', 'totalNSFV'])
		df_prev['ns'] = 1 - df_prev.sum(axis=1)
		df_prev = AgeToCohorts(df_prev, self.cohortSize, self.cohortOffset)
		OutputToFile(df_prev, path_out + fileName)

	def ProcessRelativeRisk(self):
		path = self.data_dir + 'rr_disease/'
		path_out = self.data_post_dir + 'rr_disease/'

		rr_suffix = '_rr.csv'
		p_rr = pathlib.Path(path)
		rr_paths = sorted(p_rr.glob('*{}'.format(rr_suffix)))
		for rr_path in rr_paths:
			disease = str(rr_path.name)[:-len(rr_suffix)]
			print(disease)
			df = SetStandardIndex(AddStrata(pd.read_csv(rr_path).set_index(['age', 'sex'])))
			df['ageval'] = df.index.get_level_values('age')
			df_param = pd.read_csv(path + '{}_params.csv'.format(disease))
			
			Smoker_C = df_param['Smoker_C'][0]
			Smoker_N = df_param['Smoker_N'][0]
			Vaper_C = df_param['Vaper_C'][0]
			Vaper_N = df_param['Vaper_N'][0]
			DecayAge = df_param['DecayAge'][0]

			for i in range(1, self.tunnel_length):
				df['qsqv_{}'.format(i)] = 1 + (df['cs'] - 1) * np.exp(
					-i * Smoker_C * np.exp(
						-Smoker_N * (np.maximum(0, df['ageval'] - DecayAge))))
				df['qscv_{}'.format(i)] = df['nscv'] + (df['cscv'] - df['nscv']) * np.exp(
					-i * Smoker_C * np.exp(
						-Smoker_N * (np.maximum(0, df['ageval'] - DecayAge))))
				df['nsqv_{}'.format(i)] = 1 + (df['nscv'] - 1) * np.exp(
					-i * Vaper_C * np.exp(
						-Vaper_N * (np.maximum(0, df['ageval'] - DecayAge))))

			df['fsfv'] = 1
			df['fscv'] = df['nscv']
			df['nsfv'] = 1
			df['ns'] = 1
			df = df.drop(columns=['ageval'])
			OutputToFile(df, path_out + '{}_rr'.format(disease))
