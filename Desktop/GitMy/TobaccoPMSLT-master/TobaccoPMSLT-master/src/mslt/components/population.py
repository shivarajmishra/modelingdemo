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

def load_population_data(builder):
	pop_data = builder.data.load('population.structure')
	pop_data['age'] = pop_data['age'].astype(float)
	pop_data = pop_data[['age', 'sex', 'strata', 'value']].rename(columns={'value': 'population'})
	pop_data['population'] = pop_data['population'].astype(float)
	pop_data['bau_population'] = pop_data['population']

	if 'bucket_width' in builder.configuration.population:
		bucket_width = builder.configuration.population.bucket_width
		bucket_offset = builder.configuration.population.bucket_offset
		bucket_end = builder.configuration.population.max_age + bucket_offset
		pop_data = pop_data.set_index('age')
		pop_data = pop_data.loc[[float(x) for x in range(bucket_offset, bucket_end, bucket_width)], :]
		pop_data = pop_data.reset_index()
	
	return pop_data


class BasePopulation:
	"""
	This component implements the core population demographics: age, sex, strata,
	population size.

	The configuration options for this component are:

	``population_size``
		The number of population cohorts (**must be specified**).
	``max_age``
		The age at which cohorts are removed from the population
		(default: 110).

	.. code-block:: yaml

	   configuration
		   population:
			   population_size: 44 # Male and female 5-year cohorts, 0 to 109.
			   max_age: 110        # The age at which cohorts are removed.

	"""

	configuration_defaults = {
		'population': {
			'max_age': 110,
		}
	}

	@property
	def name(self):
		return 'base_population'
	
	def setup(self, builder):
		"""Load the population data."""
		columns = ['age', 'sex', 'strata', 'population', 'bau_population',
				   'acmr', 'bau_acmr',
				   'pr_death', 'bau_pr_death', 'deaths', 'bau_deaths',
				   'yld_rate', 'bau_yld_rate',
				   'expenditure_rate', 'bau_expenditure_rate',
				   'expenditure_rate_death', 'bau_expenditure_rate_death',
				   'total_spent', 'bau_total_spent',
				   'income', 'bau_income',
				   'income_death', 'bau_income_death',
				   'total_income', 'bau_total_income',
				   'alive_person_years', 'bau_alive_person_years',
				   'dead_person_years', 'bau_dead_person_years',
				   'HALY', 'bau_HALY']

		self.pop_data = load_population_data(builder)
		
		# Create additional columns with placeholder (zero) values.
		for column in columns:
			if column in self.pop_data.columns:
				continue
			self.pop_data.loc[:, column] = 0.0

		self.max_age = builder.configuration.population.max_age

		if 'strata' in builder.configuration.population:
			util.SetStrataDf(pd.DataFrame({'strata' : builder.configuration.population.strata}))

		start_year = builder.configuration.time.start.year
		start_month = builder.configuration.time.start.month
		start_day = builder.configuration.time.start.day

		self.start_date = date(year=start_year,
							   month=start_month,
							   day=start_day)

		self.clock = builder.time.clock()
		#Denominator is 365.25 to Account for leap years in aging
		self.years_per_timestep = builder.configuration.time.step_size/365.25

		# Track all of the quantities that exist in the core spreadsheet table.
		builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=columns)
		self.population_view = builder.population.get_view(columns + ['tracked'])

		# Age cohorts before each time-step (except the first time-step).
		builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)

	def on_initialize_simulants(self, _):
		"""Initialize each cohort."""
		self.population_view.update(self.pop_data)

	def on_time_step_prepare(self, event):
		"""Remove cohorts that have reached the maximum age."""
		pop = self.population_view.get(event.index, query='tracked == True')
		# Don't do this
		## Only increase cohort ages after the first time-step.
		##if self.clock().date() > self.start_date:
		pop['age'] += self.years_per_timestep
		pop.loc[pop.age > self.max_age, 'tracked'] = False
		self.population_view.update(pop)


class Mortality:
	"""
	This component reduces the population size of each cohort over time,
	according to the all-cause mortality rate.
	"""
	
	@property
	def name(self):
		return 'mortality'

	def setup(self, builder):
		"""Load the all-cause mortality rate."""
		mortality_data = builder.data.load('cause.all_causes.mortality')
		self.mortality_rate = builder.value.register_value_producer(
			'mortality_rate', source=builder.lookup.build_table(mortality_data, 
																key_columns=['sex', 'strata'], 
																parameter_columns=['age','year']))

		self.bau_mortality_rate = builder.value.register_value_producer(
			'bau_mortality_rate', source=builder.lookup.build_table(mortality_data, 
																	key_columns=['sex', 'strata'], 
																	parameter_columns=['age','year']))

		self.years_per_timestep = builder.configuration.time.step_size/365
		builder.event.register_listener('time_step', self.on_time_step)

		self.population_view = builder.population.get_view([
			'population', 'bau_population', 'acmr', 'bau_acmr',
			'pr_death', 'bau_pr_death', 'deaths', 'bau_deaths',
			'alive_person_years', 'bau_alive_person_years',
			'dead_person_years', 'bau_dead_person_years',	
		])

	def on_time_step(self, event):
		"""
		Calculate the number of deaths and survivors at each time-step, for
		both the BAU and intervention scenarios.
		"""
		pop = self.population_view.get(event.index)
		if pop.empty:
			return
		
		pop.acmr = self.mortality_rate(event.index)
		pop.pr_death = 1 - np.exp(-pop.acmr * self.years_per_timestep)
		pop.deaths = pop.population * pop.pr_death
		pop.population *= 1 - pop.pr_death
		pop.alive_person_years = pop.population * self.years_per_timestep
		pop.dead_person_years = pop.deaths * self.years_per_timestep

		pop.bau_acmr = self.bau_mortality_rate(event.index)
		pop.bau_pr_death = 1 - np.exp(-pop.bau_acmr * self.years_per_timestep)
		pop.bau_deaths = pop.bau_population * pop.bau_pr_death
		pop.bau_population *= 1 - pop.bau_pr_death
		pop.bau_alive_person_years = pop.bau_population * self.years_per_timestep
		pop.bau_dead_person_years = pop.bau_deaths * self.years_per_timestep

		self.population_view.update(pop)


class MortalityEffects:
	"""
	This component adjusts the mortality rate based on external inputs.
	"""
	
	def __init__(self, name):
		self._name = name

	@property
	def name(self):
		return f'{self._name}_mort_effects'

	def setup(self, builder):
		self.years_per_timestep = builder.configuration.time.step_size/365

		mort_effects_data = builder.data.load(f'mortality_effects.{self._name}')
		self.mort_effects_table = builder.lookup.build_table(mort_effects_data, 
														key_columns=['sex', 'strata'],
														parameter_columns=['age','year'])

		self.register_mortality_modifier(builder)


	def register_mortality_modifier(self, builder):
		rate_name = 'mortality_rate'
		modifier = lambda ix, mort_rate: self.mortality_rate_adjustment(ix, mort_rate)
		builder.value.register_value_modifier(rate_name, modifier)


	def mortality_rate_adjustment(self, index, mort_rate):
		mort_scale = self.mort_effects_table(index)
		new_rate = mort_rate * mort_scale

		return new_rate


class Disability:
	"""
	This component calculates the health-adjusted life years (HALYs) for each
	cohort over time, according to the years lost due to disability (YLD)
	rate.
	"""
	
	@property
	def name(self):
		return 'disability'

	def setup(self, builder):
		"""Load the years lost due to disability (YLD) rate."""
		yld_data = builder.data.load('cause.all_causes.disability_rate')
		yld_rate = builder.lookup.build_table(yld_data, 
											  key_columns=['sex', 'strata'], 
											  parameter_columns=['age','year'])
		self.yld_rate = builder.value.register_value_producer('yld_rate', source=yld_rate)
		self.bau_yld_rate = builder.value.register_value_producer('bau_yld_rate', source=yld_rate)

		self.years_per_timestep = builder.configuration.time.step_size/365
		builder.event.register_listener('time_step', self.on_time_step)

		self.population_view = builder.population.get_view([
			'bau_yld_rate', 'yld_rate',
			'bau_alive_person_years', 'alive_person_years',
			'bau_dead_person_years', 'dead_person_years',
			'bau_HALY', 'HALY'])

	def on_time_step(self, event):
		"""
		Calculate the HALYs for each cohort at each time-step, for both the
		BAU and intervention scenarios.
		"""
		pop = self.population_view.get(event.index)
		if pop.empty:
			return
		pop.yld_rate = self.yld_rate(event.index)
		pop.bau_yld_rate = self.bau_yld_rate(event.index)
		# Rescale yld_rate to per year, person_years is already person years per timestep.
		pop.HALY = (pop.alive_person_years + 0.5 * pop.dead_person_years) * (1 - pop.yld_rate)
		pop.bau_HALY = (pop.bau_alive_person_years + 0.5 * pop.bau_dead_person_years) * (1 - pop.bau_yld_rate)
		self.population_view.update(pop)


class Expenditure:
	"""
	This component calculates the health expendtiure for each
	cohort over time.
	"""

	@property
	def name(self):
		return 'expenditure'

	def setup(self, builder):
		"""Load the expenditure rate."""
		expenditure_data = builder.data.load('cause.all_causes.expenditure_rate')
		expenditure_rate = builder.lookup.build_table(expenditure_data, 
											  key_columns=['sex', 'strata'], 
											  parameter_columns=['age','year'])
		self.expenditure_rate = builder.value.register_value_producer('expenditure_rate', source=expenditure_rate)
		self.bau_expenditure_rate = builder.value.register_value_producer('bau_expenditure_rate', source=expenditure_rate)

		"""Load the expenditure death cost."""
		death_data = builder.data.load('cause.all_causes.expenditure_rate_death')
		death_cost = builder.lookup.build_table(death_data, 
											  key_columns=['sex', 'strata'], 
											  parameter_columns=['age','year'])
		self.expenditure_rate_death = builder.value.register_value_producer('expenditure_rate_death', source=death_cost)
		self.bau_expenditure_rate_death = builder.value.register_value_producer('bau_expenditure_rate_death', source=death_cost)

		self.years_per_timestep = builder.configuration.time.step_size/365
		builder.event.register_listener('time_step', self.on_time_step)

		self.population_view = builder.population.get_view([
			'bau_population', 'population',
			'bau_expenditure_rate', 'expenditure_rate',
			'bau_expenditure_rate_death', 'expenditure_rate_death',
			'bau_alive_person_years', 'alive_person_years',
			'bau_dead_person_years', 'dead_person_years',
			'deaths', 'bau_deaths',
			'total_spent', 'bau_total_spent',])

	def on_time_step(self, event):
		"""
		Calculate the total spent for each cohort at each time-step, for both the
		BAU and intervention scenarios.
		"""
		pop = self.population_view.get(event.index)
		if pop.empty:
			return
		pop.expenditure_rate = self.expenditure_rate(event.index)
		pop.bau_expenditure_rate = self.bau_expenditure_rate(event.index)
		pop.expenditure_rate_death = self.expenditure_rate_death(event.index)
		pop.bau_expenditure_rate_death = self.bau_expenditure_rate_death(event.index)
		# Split expenditure into people who live through the timestep and people who die in the timestep.
		pop.total_spent = (pop.alive_person_years * self.expenditure_rate(event.index) + 
			pop.dead_person_years * self.expenditure_rate_death(event.index))
		pop.bau_total_spent = (pop.bau_alive_person_years * self.bau_expenditure_rate(event.index) + 
			pop.bau_dead_person_years * self.bau_expenditure_rate_death(event.index))
		self.population_view.update(pop)


class Income:
	"""
	This component calculates the income for each
	cohort over time.
	"""

	@property
	def name(self):
		return 'income'

	def setup(self, builder):
		"""Load the income rate."""
		income_data = builder.data.load('cause.all_causes.income')
		income_rate = builder.lookup.build_table(income_data, 
											  key_columns=['sex', 'strata'], 
											  parameter_columns=['age','year'])
		self.income = builder.value.register_value_producer('income', source=income_rate)
		self.bau_income = builder.value.register_value_producer('bau_income', source=income_rate)

		income_death_data = builder.data.load('cause.all_causes.income_death')
		income_death_rate = builder.lookup.build_table(income_death_data, 
											  key_columns=['sex', 'strata'], 
											  parameter_columns=['age','year'])
		self.income_death = builder.value.register_value_producer('income_death', source=income_rate)
		self.bau_income_death = builder.value.register_value_producer('bau_income_death', source=income_rate)

		self.years_per_timestep = builder.configuration.time.step_size/365
		builder.event.register_listener('time_step', self.on_time_step)

		self.population_view = builder.population.get_view([
			'bau_population', 'population',
			'bau_income', 'income',
			'bau_income_death', 'income_death',
			'bau_total_income', 'total_income',
			'bau_alive_person_years', 'alive_person_years',
			'bau_dead_person_years', 'dead_person_years',
		])

	def on_time_step(self, event):
		"""
		Calculate the total income for each cohort at each time-step, for both the
		BAU and intervention scenarios.
		"""
		pop = self.population_view.get(event.index)
		if pop.empty:
			return
		pop.income = self.income(event.index)
		pop.bau_income = self.bau_income(event.index)
		pop.income_death = self.income_death(event.index)
		pop.bau_income_death = self.bau_income_death(event.index)
		# Split income into people who live through the timestep and people who die in the timestep.
		pop.total_income = (pop.alive_person_years * self.income(event.index) + 
			pop.dead_person_years * self.income_death(event.index))
		pop.bau_total_income = (pop.bau_alive_person_years * self.bau_income(event.index) + 
			pop.bau_dead_person_years * self.bau_income_death(event.index))
		self.population_view.update(pop)
