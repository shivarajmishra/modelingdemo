"""
=========
Observers
=========

This module contains tools for recording various outputs of interest in
multi-state lifetable simulations.

"""
import numpy as np
import pandas as pd
import os
from datetime import date

from .circuit import GetStateCol


def MakePath(path):
	if '/' not in path:
		return
	out_folder = os.path.dirname(path)
	if not os.path.exists(out_folder):
		MakePath(out_folder)
		os.mkdir(out_folder)


def output_csv_mkdir(data, path, index):
	"""
	Wrapper for pandas .to_csv() method to create directory for path if it
	doesn't already exist.
	"""
	MakePath(path)
	data.to_csv(path, index=index)


def output_file(config, suffix, sep='_', ext='csv'):
	"""
	Determine the output file name for an observer, based on the prefix
	defined in ``config.observer.output_prefix`` and the (optional)
	``config.input_data.input_draw_number``.

	Parameters
	----------
	config
		The builder configuration object.
	suffix
		The observer-specific suffix.
	sep
		The separator between prefix, suffix, and draw number.
	ext
		The output file extension.

	"""
	if 'observer' not in config:
		raise ValueError('observer.output_prefix not defined')
	if 'output_prefix' not in config.observer:
		raise ValueError('observer.output_prefix not defined')
	prefix = config.observer.output_prefix
	if 'input_draw_number' in config.input_data:
		draw = config.input_data.input_draw_number
	else:
		draw = 0
	out_file = prefix + sep + suffix
	if draw > 0:
		out_file += '{}{}'.format(sep, draw)
	out_file += '.{}'.format(ext)
	return out_file


class MorbidityMortality:
	"""
	This class records the all-cause morbidity and mortality rates for each
	cohort at each year of the simulation.

	Parameters
	----------
	output_suffix
		The suffix for the CSV file in which to record the
		morbidity and mortality data.

	"""

	def __init__(self, output_suffix='mm'):
		self.output_suffix = output_suffix

	@property
	def name(self):
		return 'morbidity_mortality_observer'

	def setup(self, builder):
		# Record the key columns from the core multi-state life table.
		columns = ['age', 'sex', 'strata',
				   'population', 'bau_population',
				   'acmr', 'bau_acmr',
				   'pr_death', 'bau_pr_death',
				   'deaths', 'bau_deaths',
				   'yld_rate', 'bau_yld_rate',
				   'dead_person_years', 'bau_dead_person_years',
				   'alive_person_years', 'bau_alive_person_years',
				   'expenditure_rate', 'bau_expenditure_rate',
				   'total_spent', 'bau_total_spent',
				   'income', 'bau_income',
				   'total_income', 'bau_total_income',
				   'HALY', 'bau_HALY']
		self.population_view = builder.population.get_view(columns)
		self.clock = builder.time.clock()

		# Output the start of year 0
		start_year = builder.configuration.time.start.year
		start_month = builder.configuration.time.start.month
		start_day = builder.configuration.time.start.day
		self.start_date = date(year=start_year,
							   month=start_month,
							   day=start_day)
		builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)

		builder.event.register_listener('collect_metrics', self.on_collect_metrics)
		builder.event.register_listener('simulation_end', self.write_output)
		self.tables = []
		self.table_cols = ['sex', 'age', 'strata', 'year', 'month',
						   'population', 'bau_population',
						   'prev_population', 'bau_prev_population',
						   'acmr', 'bau_acmr',
						   'pr_death', 'bau_pr_death',
						   'deaths', 'bau_deaths',
						   'yld_rate', 'bau_yld_rate',
						   'dead_person_years', 'bau_dead_person_years',
						   'alive_person_years', 'bau_alive_person_years',
						   'expenditure_rate', 'bau_expenditure_rate',
						   'total_spent', 'bau_total_spent',
						   'income', 'bau_income',
				   			'total_income', 'bau_total_income',
						   'HALY', 'bau_HALY']

		self.output_file = output_file(builder.configuration,
									   self.output_suffix)

	def on_collect_metrics(self, event):
		pop = self.population_view.get(event.index)
		if len(pop.index) == 0:
			# No tracked population remains.
			return

		pop['year'] = self.clock().year
		pop['month'] = self.clock().month
		# Record the population size prior to the deaths.
		pop['prev_population'] = pop['population'] + pop['deaths']
		pop['bau_prev_population'] = pop['bau_population'] + pop['bau_deaths']
		self.tables.append(pop[self.table_cols])

	
	def on_time_step_prepare(self, event):
		# Only output the start of year 0
		if self.clock().date() != self.start_date:
			return
		pop = self.population_view.get(event.index)
		if len(pop.index) == 0:
			# No tracked population remains.
			return

		pop['year'] = self.clock().year - 1
		pop['month'] = self.clock().month
		# Record the population size prior to the deaths.
		pop['prev_population'] = pop['population'] + pop['deaths']
		pop['bau_prev_population'] = pop['bau_population'] + pop['bau_deaths']
		self.tables.append(pop[self.table_cols])


	def calculate_LE(self, table, py_col, denom_col):
		"""Calculate the life expectancy for each cohort at each time-step.

		Parameters
		----------
		table
			The population life table.
		py_col
			The name of the person-years column.
		denom_col
			The name of the population denominator column.

		Returns
		-------
			The life expectancy for each table row, represented as a
			pandas.Series object.

		"""
		# Group the person-years by cohort.
		group_cols = ['year_of_birth', 'sex', 'strata']
		subset_cols = group_cols + [py_col]
		grouped = table.loc[:, subset_cols].groupby(by=group_cols)[py_col]
		# Calculate the reverse-cumulative sums of the adjusted person-years
		# (i.e., the present and future person-years) by:
		#   (a) reversing the adjusted person-years values in each cohort;
		#   (b) calculating the cumulative sums in each cohort; and
		#   (c) restoring the original order.
		cumsum = grouped.apply(lambda x: pd.Series(x[::-1].cumsum()).iloc[::-1])
		return (cumsum / table[denom_col]).replace({np.inf : 0, np.nan : 0})

	def write_output(self, event):
		data = pd.concat(self.tables, ignore_index=True)
		data['year_of_birth'] = data['year'] - np.floor(data['age'])
		# Sort the table by cohort (i.e., generation and sex), and then by
		# calendar year, so that results are output in the same order as in
		# the spreadsheet models.
		data = data.sort_values(by=['year_of_birth', 'sex', 'strata', 'age', 'year', 'month'], axis=0)
		data = data.reset_index(drop=True)
		# Re-order the table columns.
		cols = ['year_of_birth'] + self.table_cols
		data = data[cols]
		# Calculate life expectancy and HALE for the BAU and intervention,
		# with respect to the initial population, not the survivors.
		data['person_years'] = data['alive_person_years'] + 0.5 * data['dead_person_years']
		data['bau_person_years'] = data['bau_alive_person_years'] + 0.5 * data['bau_dead_person_years']

		data['LE'] = self.calculate_LE(data, 'person_years', 'prev_population')
		data['bau_LE'] = self.calculate_LE(data, 'bau_person_years',
										   'bau_prev_population')
		data['HALE'] = self.calculate_LE(data, 'HALY', 'prev_population')
		data['bau_HALE'] = self.calculate_LE(data, 'bau_HALY',
										   'bau_prev_population')
		output_csv_mkdir(data, self.output_file, index=False)


class AcuteDisease:
	"""
	This class records the disease incidence rate and disease prevalence for
	each cohort at each year of the simulation.

	Parameters
	----------
	name
		The name of the chronic disease.
	output_suffix
		The suffix for the CSV file in which to record the
		disease data.

	"""

	def __init__(self, name, output_suffix=None):
		self._name = name
		if output_suffix is None:
			output_suffix = name.lower()
		self.output_suffix = output_suffix[0:11] # Very long paths break things.
		
	@property
	def name(self):
		return f'{self._name}_observer'

	def setup(self, builder):
		self.metric_deaths = self._name + '_deaths'
		self.metric_HALY = self._name + '_HALY'
		columns = ['age', 'sex', 'strata',
				   self.metric_deaths + '_bau',
				   self.metric_HALY + '_bau',
				   self.metric_deaths,
				   self.metric_HALY]
		self.population_view = builder.population.get_view(columns)

		# Output the start of year 0
		start_year = builder.configuration.time.start.year
		start_month = builder.configuration.time.start.month
		start_day = builder.configuration.time.start.day
		self.start_date = date(year=start_year,
							   month=start_month,
							   day=start_day)
		builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)

		# Output for the end of other years
		builder.event.register_listener('collect_metrics', self.on_collect_metrics)
		builder.event.register_listener('simulation_end', self.write_output)

		self.tables = []
		self.table_cols = ['sex', 'age', 'strata', 'year', 'month',
							self.metric_deaths + '_bau',
							self.metric_HALY + '_bau',
							self.metric_deaths,
							self.metric_HALY]
		self.clock = builder.time.clock()
		self.output_file = output_file(builder.configuration,
									   self.output_suffix)

	def on_collect_metrics(self, event):
		pop = self.population_view.get(event.index)
		if len(pop.index) == 0:
			# No tracked population remains.
			return

		pop['year'] = self.clock().year
		pop['month'] = self.clock().month
		self.tables.append(pop.loc[:, self.table_cols])

	def on_time_step_prepare(self, event):
		# Only output the start of year 0
		if self.clock().date() != self.start_date:
			return
		pop = self.population_view.get(event.index)
		if len(pop.index) == 0:
			# No tracked population remains.
			return

		pop['year'] = self.clock().year - 1
		pop['month'] = self.clock().month
		self.tables.append(pop.loc[:, self.table_cols])
	
	def write_output(self, event):
		data = pd.concat(self.tables, ignore_index=True)
		output_csv_mkdir(data, self.output_file, index=False)


class Disease:
	"""
	This class records the disease incidence rate and disease prevalence for
	each cohort at each year of the simulation.

	Parameters
	----------
	name
		The name of the chronic disease.
	output_suffix
		The suffix for the CSV file in which to record the
		disease data.

	"""

	def __init__(self, name, output_suffix=None):
		self._name = name
		if output_suffix is None:
			output_suffix = name.lower()
		self.output_suffix = output_suffix
		
	@property
	def name(self):
		return f'{self._name}_observer'

	def setup(self, builder):
		bau_incidence_value = '{}.incidence'.format(self._name)
		int_incidence_value = '{}_intervention.incidence'.format(self._name)
		self.bau_incidence = builder.value.get_value(bau_incidence_value)
		self.int_incidence = builder.value.get_value(int_incidence_value)

		self.bau_S_col = '{}_S'.format(self._name)
		self.bau_C_col = '{}_C'.format(self._name)
		self.int_S_col = '{}_S_intervention'.format(self._name)
		self.int_C_col = '{}_C_intervention'.format(self._name)

		columns = ['age', 'sex', 'strata',
				   self.bau_S_col, self.bau_C_col,
				   self.int_S_col, self.int_C_col]
		self.population_view = builder.population.get_view(columns)

		# Output the start of year 0
		start_year = builder.configuration.time.start.year
		start_month = builder.configuration.time.start.month
		start_day = builder.configuration.time.start.day
		self.start_date = date(year=start_year,
							   month=start_month,
							   day=start_day)
		builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)

		# Output for the end of other years
		builder.event.register_listener('collect_metrics', self.on_collect_metrics)
		builder.event.register_listener('simulation_end', self.write_output)

		self.tables = []
		self.table_cols = ['sex', 'age', 'strata', 'year',
						   'bau_incidence', 'int_incidence',
						   'bau_prevalence', 'int_prevalence',
						   'bau_deaths', 'int_deaths']
		self.clock = builder.time.clock()
		self.output_file = output_file(builder.configuration,
									   self.output_suffix)

	def on_collect_metrics(self, event):
		pop = self.population_view.get(event.index)
		if len(pop.index) == 0:
			# No tracked population remains.
			return

		pop['year'] = self.clock().year
		pop['bau_incidence'] = self.bau_incidence(event.index)
		pop['int_incidence'] = self.int_incidence(event.index)
		pop['bau_prevalence'] = pop[self.bau_C_col] / (pop[self.bau_C_col] + pop[self.bau_S_col])
		pop['int_prevalence'] = pop[self.int_C_col] / (pop[self.bau_C_col] + pop[self.bau_S_col])
		pop['bau_deaths'] = 1000 - pop[self.bau_S_col] - pop[self.bau_C_col]
		pop['int_deaths'] = 1000 - pop[self.int_S_col] - pop[self.int_C_col]
		self.tables.append(pop.loc[:, self.table_cols])

	def on_time_step_prepare(self, event):
		# Only output the start of year 0
		if self.clock().date() != self.start_date:
			return
		pop = self.population_view.get(event.index)
		if len(pop.index) == 0:
			# No tracked population remains.
			return

		pop['year'] = self.clock().year - 1
		pop['bau_incidence'] = self.bau_incidence(event.index)
		pop['int_incidence'] = self.int_incidence(event.index)
		pop['bau_prevalence'] = pop[self.bau_C_col] / (pop[self.bau_C_col] + pop[self.bau_S_col])
		pop['int_prevalence'] = pop[self.int_C_col] / (pop[self.bau_C_col] + pop[self.bau_S_col])
		pop['bau_deaths'] = 1000 - pop[self.bau_S_col] - pop[self.bau_C_col]
		pop['int_deaths'] = 1000 - pop[self.int_S_col] - pop[self.int_C_col]
		self.tables.append(pop.loc[:, self.table_cols])

	def write_output(self, event):
		data = pd.concat(self.tables, ignore_index=True)
		data['diff_incidence'] = data['int_incidence'] - data['bau_incidence']
		data['diff_prevalence'] = data['int_prevalence'] - data['bau_prevalence']
		data['year_of_birth'] = data['year'] - data['age']
		data['disease'] = self._name
		# Sort the table by cohort (i.e., generation and sex), and then by
		# calendar year, so that results are output in the same order as in
		# the spreadsheet models.
		data = data.sort_values(by=['year_of_birth', 'sex', 'age', 'strata'], axis=0)
		data = data.reset_index(drop=True)
		# Re-order the table columns.
		diff_cols = ['diff_incidence', 'diff_prevalence']
		cols = ['disease', 'year_of_birth'] + self.table_cols + diff_cols
		data = data[cols]
		output_csv_mkdir(data, self.output_file, index=False)


class Circuit:
	"""
	This class records the state of the intervention and BAU circuit.

	Parameters
	----------
	name
		The name of the circuit.
	output_suffix
		The suffix for the CSV file in which to record the
		disease data.

	"""

	def __init__(self, output_suffix=None):
		self._name = 'circuit'
		if output_suffix is None:
			output_suffix = self._name.lower()
		self.output_suffix = output_suffix[0:11] # Very long paths break things.
		
	@property
	def name(self):
		return f'{self._name}_observer'

	def setup(self, builder):
		stateCols = []
		"""Configuration."""
		if 'circuit' in builder.configuration:
			# Determine which states exist
			if 'arcs' in builder.configuration.circuit:
				for source in builder.configuration.circuit.arcs:
					stateCols.append(GetStateCol(source))
					stateCols.append(GetStateCol(source, bau=True))
		
		"""Listener and Outputs"""
		self.population_view = builder.population.get_view(['age', 'sex', 'strata'] + stateCols)

		# Output the start of year 0
		start_year = builder.configuration.time.start.year
		start_month = builder.configuration.time.start.month
		start_day = builder.configuration.time.start.day
		self.start_date = date(year=start_year,
							   month=start_month,
							   day=start_day)
		builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)

		# Output for the end of other years
		builder.event.register_listener('collect_metrics', self.on_collect_metrics)
		builder.event.register_listener('simulation_end', self.write_output)

		self.tables = []
		self.table_cols = ['sex', 'age', 'strata', 'year', 'month'] + stateCols
		self.clock = builder.time.clock()
		self.output_file = output_file(builder.configuration, self.output_suffix)

	def on_collect_metrics(self, event):
		pop = self.population_view.get(event.index)
		if len(pop.index) == 0:
			# No tracked population remains.
			return

		pop['year'] = self.clock().year
		pop['month'] = self.clock().month
		self.tables.append(pop.loc[:, self.table_cols])

	def write_output(self, event):
		data = pd.concat(self.tables, ignore_index=True)

		data['year_of_birth'] = data['year'] - data['age']
		# Sort the table by cohort (i.e., generation and sex), and then by
		# calendar year, so that results are output in the same order as in
		# the spreadsheet models.
		data = data.sort_values(by=['year_of_birth', 'sex', 'age', 'strata'], axis=0)
		data = data.reset_index(drop=True)
		# Re-order the table columns.
		cols = ['year_of_birth'] + self.table_cols
		data = data.reindex(columns=cols)

		output_csv_mkdir(data, self.output_file, index=False)

	def on_time_step_prepare(self, event):
		# Only output the start of year 0
		if self.clock().date() != self.start_date:
			return
		pop = self.population_view.get(event.index)
		if len(pop.index) == 0:
			# No tracked population remains.
			return
		pop['year'] = self.clock().year - 1
		pop['month'] = self.clock().month
		self.tables.append(pop.loc[:, self.table_cols])
