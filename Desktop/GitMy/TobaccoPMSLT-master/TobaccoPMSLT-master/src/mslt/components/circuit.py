"""
==================
Demographic Models
==================

This module contains tools for modeling the core demography in
multi-state lifetable simulations.

"""
import numpy as np
import pandas as pd
from datetime import date

import mslt.utilities as util

def IsInArtifact(artifact, name):
	try:
		artifact.load(name)
	except:
		return False
	return True


def GetStateCol(state, bau=False):
	if bau:
		return 'c_{}_bau'.format(state)
	return 'c_{}'.format(state)


class Circuit:
	"""
	This component implements a compartment circuit model.

	The configuration options for this component are:

	``arcs``
		The existing arcs.

	"""

	@property
	def name(self):
		return 'circuit'
	
	def setup(self, builder):
		self.arcs = []
		self.static_arcs = []
		self.states = {}
		self.columns = []
		self.cols_int = []
		self.cols_bau = []
		self.disease_rr = {}
		self.rr_ignore = {}
		self.prevelanceName_int = 'prevalance'
		self.prevelanceName_bau = 'prevalance'

		"""Configuration."""
		if 'circuit' in builder.configuration:
			# Intervention and BAU can start with differing prevalence
			if 'prevalence_int' in builder.configuration.circuit:
				self.prevelanceName_int = builder.configuration.circuit.prevalence_int
			if 'prevalence_bau' in builder.configuration.circuit:
				self.prevelanceName_bau = builder.configuration.circuit.prevalence_bau

			# Determine which arcs to include, and register them.
			if 'arcs' in builder.configuration.circuit:
				for source in builder.configuration.circuit.arcs:
					sourceName = GetStateCol(source)
					sourceName_bau = GetStateCol(source, bau=True)

					self.states[source] = True
					self.columns.append(sourceName)
					self.columns.append(sourceName_bau)
					self.cols_int.append(sourceName)
					self.cols_bau.append(sourceName_bau)

					for sink in builder.configuration.circuit.arcs[source]:
						sinkName = 'c_{}'.format(sink)
						sinkName_bau = 'c_{}_bau'.format(sink)

						flowName = 'circuit.flow.{}_{}'.format(source, sink)
						flowName_bau = 'circuit.flow_bau.{}_{}'.format(source, sink)

						flowTable = builder.data.load(flowName)
						# Only register a value producer for arcs that have magic wands.
						if flowName in builder.configuration.magic_wand_flow_register:
							flowRate = builder.lookup.build_table(
								flowTable,
								key_columns=['sex', 'strata'], 
								parameter_columns=['age','year'])

							self.arcs.append({
								'source' : sourceName,
								'sink' : sinkName,
								'rate' : builder.value.register_value_producer(
									flowName, source=flowRate),
								'source_bau' : sourceName_bau,
								'sink_bau' : sinkName_bau,
								'rate_bau' : builder.value.register_value_producer(
									flowName_bau, source=flowRate),
							})
						else:
							self.static_arcs.append({
								'source' : sourceName,
								'sink' : sinkName,
								'static_rate_name' : flowName,
								'source_bau' : sourceName_bau,
								'sink_bau' : sinkName_bau,
								'rate_both' : builder.lookup.build_table(
									flowTable, 
									key_columns=['sex', 'strata'], parameter_columns=['age','year'])
							})
			
			# Get the states that are ignored by the rr calculation. This will usually be the dead ones.
			if 'rr_ignore' in builder.configuration.circuit:
				for state in builder.configuration.circuit.rr_ignore:
					stateName = GetStateCol(state)
					stateName_bau = GetStateCol(state, bau=True)
					self.rr_ignore[stateName] = True
					self.rr_ignore[stateName_bau] = True
			
			# Register a modifier for each disease affected by the circuit.
			registerRR = (('register_rr' in builder.configuration.circuit) and builder.configuration.circuit.register_rr)
			if 'affects' in builder.configuration.circuit:
				for disease in builder.configuration.circuit.affects:
					self.register_modifier(builder, disease)
					self.disease_rr[disease] = {}
					rr_data = False
					for state in self.states.keys():
						if GetStateCol(state) not in self.rr_ignore:
							rr_col = builder.data.load('circuit.rr.{}_{}'.format(disease, state))
							if type(rr_data) == bool:
								rr_data = rr_col
							rr_data[GetStateCol(state)] = rr_col['value']
					
					rr_data = rr_data.drop(columns=['value'])
					rr_table = builder.lookup.build_table(
						rr_data, 
						key_columns=['sex', 'strata'], 
						parameter_columns=['age','year'])
					
					self.disease_rr[disease] = rr_table
		
		"""Circuit data"""
		self.state_data = self.load_circuit(builder)
		self.state_view = builder.population.get_view(self.columns)
		builder.population.initializes_simulants(self.on_initialize, creates_columns=self.columns)

		builder.event.register_listener('time_step', self.on_time_step)
		

	def load_circuit(self, builder):
		df_states = pd.DataFrame()
		for state in self.states:
			# Load intervention prevalence
			df = builder.data.load('circuit.{}.{}'.format(self.prevelanceName_int, state))
			df['age'] = df['age'].astype(float)
			df['value'] = df['value'].astype(float)
			df = df[['age', 'sex', 'strata', 'value']].rename(
				columns={'value': 'c_{}'.format(state)})
			df = df.set_index(['age', 'sex', 'strata']).sort_index()
			df_states[['c_{}'.format(state)]] = df
			
			# Load bau prevalence
			df = builder.data.load('circuit.{}.{}'.format(self.prevelanceName_bau, state))
			df['age'] = df['age'].astype(float)
			df['value'] = df['value'].astype(float)
			df = df[['age', 'sex', 'strata', 'value']].rename(
				columns={'value': 'c_{}_bau'.format(state)})
			df = df.set_index(['age', 'sex', 'strata']).sort_index()
			df_states[['c_{}_bau'.format(state)]] = df

		if 'bucket_width' in builder.configuration.population:
			bucket_width = builder.configuration.population.bucket_width
			bucket_offset = builder.configuration.population.bucket_offset
			bucket_end = builder.configuration.population.max_age + bucket_offset
			df_states = df_states.loc[[float(x) for x in range(bucket_offset, bucket_end, bucket_width)], :]

		return df_states


	def on_initialize(self, _):
		"""Initialize the compartments"""
		self.state_view.update(self.state_data)


	def on_time_step(self, event):
		"""
		Calculate the flow into each component
		"""
		states = self.state_view.get(event.index)
		#print(states)
		if states.empty:
			return
		states_new = states.copy()

		for arc in self.arcs:
			flow = arc['rate'](event.index)
			#print('arc', arc)
			#print('IS_NAN', flow.isnull().values.any())
			#print(flow)
			states_new[arc['sink']] += states[arc['source']] * flow
			states_new[arc['source']] -= states[arc['source']] * flow

			flow_bau = arc['rate_bau'](event.index)
			states_new[arc['sink_bau']] += states[arc['source_bau']] * flow_bau
			states_new[arc['source_bau']] -= states[arc['source_bau']] * flow_bau
			#print(states_new)

		for arc in self.static_arcs:
			flow = arc['rate_both'](event.index)
			#print('arc static', arc)
			#print('IS_NAN', flow.isnull().values.any())
			#print(flow)
			states_new[arc['sink']] += states[arc['source']] * flow
			states_new[arc['source']] -= states[arc['source']] * flow

			states_new[arc['sink_bau']] += states[arc['source_bau']] * flow
			states_new[arc['source_bau']] -= states[arc['source_bau']] * flow
			#print(states_new)

		#print(states_new)
		self.state_view.update(states_new)


	def register_modifier(self, builder, disease):
		"""Register that a disease incidence rate will be modified by this
		delayed risk in the intervention scenario.

		Parameters
		----------
		builder
			The builder object for the simulation, which provides
			access to event handlers and rate modifiers.
		disease
			The name of the disease whose incidence rate will be
			modified.

		"""
		rate_templates = []
		if IsInArtifact(builder.data, 'chronic_disease.{}.incidence'.format(disease)):
			rate_templates += ['{}_intervention.incidence']
		if IsInArtifact(builder.data, 'acute_disease.{}.mortality'.format(disease)):
			rate_templates += ['{}_intervention.excess_mortality', '{}_intervention.yld_rate']
		
		for template in rate_templates:
			rate_name = template.format(disease)
			modifier = lambda ix, rate: self.incidence_adjustment(disease, ix, rate)
			builder.value.register_value_modifier(rate_name, modifier)


	def RemoveRrIgnore(self, df):
		df['dead'] = 0
		for state in self.rr_ignore:
			if state in df.columns:
				df['dead'] += df[state]
				df = df.drop(columns=state)
		df = df.divide(1 - df['dead'], axis=0)
		df = df.drop(columns='dead')
		# Fail gracefully if everyone is dead. The rows of the df expect
		# to add to 1.
		df = df.fillna(1 / len(df.columns))
		return df


	def MakeDiseaseRrFrame(self, index, disease):
		return self.disease_rr[disease](index)


	def FindRrMean(self, index, df, df_disease, append=False):
		if append:
			df_disease = df_disease.rename(
				columns={col : col + append for col in df_disease.columns})
		df = df.mul(df_disease)
		meanSum = df.sum(axis=1)
		return meanSum


	def incidence_adjustment(self, disease, index, incidence_rate):
		"""Modify a disease incidence rate in the intervention scenario.

		Parameters
		----------
		disease
			The name of the disease.
		index
			The index into the population life table.
		incidence_rate
			The un-adjusted disease incidence rate.

		"""
		fullView = self.state_view.get(index)
		df_int = fullView.loc[:, self.cols_int]
		df_bau = fullView.loc[:, self.cols_bau]

		df_int = self.RemoveRrIgnore(df_int)
		df_bau = self.RemoveRrIgnore(df_bau)

		df_disease_rr = self.MakeDiseaseRrFrame(fullView.index, disease)

		df_int = self.FindRrMean(fullView.index, df_int, df_disease_rr)
		df_bau = self.FindRrMean(fullView.index, df_bau, df_disease_rr, append='_bau')
		df_pif = (df_bau - df_int) / df_bau
		
		return incidence_rate * (1 - df_pif)
