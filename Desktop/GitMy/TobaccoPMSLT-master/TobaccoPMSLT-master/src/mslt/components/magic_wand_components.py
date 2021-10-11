"""
=================
Magic Wand Models
=================

This module contains tools for making crude adjustments to rates in
multi-state lifetable simulations.

"""

import numpy as np
import pandas as pd

from mslt.utilities import UnstackDraw, CrossDf, OutputToFile, AgeToCohorts, SetStandardIndex
from mslt.utilities import ExpandAgeCategory, ExpandToValueAtAge, AddAge, AddSex, AddStrata, ValueToValueRange

class MortalityShift:
	
	@property
	def name(self):
		return 'mortality_shift'

	def setup(self, builder):
		builder.value.register_value_modifier('mortality_rate', self.mortality_adjustment)

	def mortality_adjustment(self, index, rates):
		return rates * .5


class YLDShift:
	
	@property
	def name(self):
		return 'yld_shift'

	def setup(self, builder):
		builder.value.register_value_modifier('yld_rate', self.disability_adjustment)

	def disability_adjustment(self, index, rates):
		return rates * .5


class IncidenceShift:

	def __init__(self, name):
		self._name = name

	@property
	def name(self):
		return self._name

	def setup(self, builder):
		builder.value.register_value_modifier(f'{self.name}.incidence', self.incidence_adjustment)

		self.rate_mult = 0.5
		"""Configuration."""
		if 'magic_wand' in builder.configuration and self.name in builder.configuration.magic_wand:
			configuration = builder.configuration.magic_wand[self.name]
			if configuration.rate_reduce:
				self.rate_mult = (1 - configuration.rate_reduce)

	def incidence_adjustment(self, index, rates):
		return rates * self.rate_mult


class AcuteIncidenceShift:

	def __init__(self, name):
		self._name = name

	@property
	def name(self):
		return self._name

	def setup(self, builder):
		self.rate_mult = 0.5
		self.track_expenditure = False
		"""Configuration."""
		if 'magic_wand' in builder.configuration and self.name in builder.configuration.magic_wand:
			configuration = builder.configuration.magic_wand[self.name]
			if configuration.rate_reduce:
				self.rate_mult = (1 - configuration.rate_reduce)
			if 'expenditure' in configuration and configuration.expenditure:
				self.track_expenditure = True
				
		builder.value.register_value_modifier(f'{self.name}.excess_mortality', self.incidence_adjustment)
		builder.value.register_value_modifier(f'{self.name}.yld_rate', self.incidence_adjustment)
		if self.track_expenditure:
			builder.value.register_value_modifier(f'{self.name}.expenditure_rate', self.incidence_adjustment)
		if self.track_income:
			builder.value.register_value_modifier(f'{self.name}.income', self.incidence_adjustment)


	def incidence_adjustment(self, index, rates):
		return rates * self.rate_mult


class GenericWand:

	def __init__(self, name):
		self._name = name

	@property
	def name(self):
		return self._name

	def setup(self, builder):
		self.rate_mult = 0
		self.rate_add = 0
		self.doSet = False
		"""Configuration."""
		if 'magic_wand' in builder.configuration and self.name in builder.configuration.magic_wand:
			configuration = builder.configuration.magic_wand[self.name]

			effect_mult = 1
			effect_add = 0
			
			if 'rate_reduce' in configuration:
				effect_mult = (1 - configuration.rate_reduce)
			if 'set_rate' in configuration:
				effect_mult = 0
				effect_add = configuration.set_rate
				self.doSet = True
			if 'add_rate' in configuration:
				effect_mult = 1
				effect_add = configuration.add_rate
				self.doSet = True
			
			if 'age_min' in configuration:
				ageRanges = []
				ageRangesAdd = []
				
				if configuration.age_min == 0:
					ageRanges.append([0, effect_mult])
					ageRangesAdd.append([0, effect_add])
				else:
					ageRanges.append([0, 1])
					ageRanges.append([configuration.age_min, effect_mult])
					ageRangesAdd.append([0, 1])
					ageRangesAdd.append([configuration.age_min, effect_add])
				
				if 'age_max' in configuration:
					ageRanges.append([configuration.age_max, 1])
					ageRangesAdd.append([configuration.age_max, effect_add])
					
				dfRate = pd.DataFrame(ageRanges,
					columns=['agecategory', 'value']
				).set_index('agecategory')

				dfAdd = pd.DataFrame(ageRanges,
					columns=['agecategory', 'value']
				).set_index('agecategory')
			else:
				dfRate = pd.DataFrame([
					[0, effect_mult]],
					columns=['agecategory', 'value']
				).set_index('agecategory')

				dfAdd = pd.DataFrame([
					[0, effect_add]],
					columns=['agecategory', 'value']
				).set_index('agecategory')

			dfRate_out = pd.DataFrame()
			dfAdd_out = pd.DataFrame()

			dfRate = SetStandardIndex(AddSex(AddStrata(
				ExpandAgeCategory(dfRate, builder.configuration.population.max_age))))
			dfRate = ValueToValueRange(dfRate, 'age', 1)

			dfAdd = SetStandardIndex(AddSex(AddStrata(
				ExpandAgeCategory(dfAdd, builder.configuration.population.max_age))))
			dfAdd = ValueToValueRange(dfAdd, 'age', 1)

			if 'strata' in configuration:
				index = list(dfAdd.index.names)

				# Very quick and dirty code below.
				dfRate = dfRate.reorder_levels([3, 0, 1, 2], axis=0).transpose()
				dfAdd = dfAdd.reorder_levels([3, 0, 1, 2], axis=0).transpose()
				tranIndex = builder.configuration.population.strata
				for strata in tranIndex:
					if strata != configuration.strata:
						dfRate[strata] = 1
						dfAdd[strata] = 0

				dfRate = dfRate.transpose().reorder_levels(index, axis=0)
				dfAdd = dfAdd.transpose().reorder_levels(index, axis=0)

			totalStart = builder.configuration.time.start.year
			totalEnd   = builder.configuration.time.start.year + builder.configuration.population.max_age

			year_start = totalStart
			year_end   = totalEnd
			if 'year_start' in configuration:
				year_start = configuration.year_start
			if 'year_end' in configuration:
				year_end = configuration.year_end

			dfDummy = dfRate.copy()

			if year_start > totalStart:
				dfDummy['year'] = totalStart
				dfDummy['value'] = 1
				dfRate_out = dfRate_out.append(
					ValueToValueRange(dfDummy, 'year', year_start - totalStart))

				dfDummy['year'] = totalStart
				dfDummy['value'] = 0
				dfAdd_out = dfAdd_out.append(
					ValueToValueRange(dfDummy, 'year', year_start - totalStart))

			dfRate['year'] = year_start
			dfRate_out = dfRate_out.append(
				ValueToValueRange(dfRate, 'year', year_end - year_start))

			dfAdd['year'] = year_start
			dfAdd_out = dfAdd_out.append(
				ValueToValueRange(dfAdd, 'year', year_end - year_start))
			
			if year_end < totalEnd:
				dfDummy['year'] = year_end
				dfDummy['value'] = 1
				dfRate_out = dfRate_out.append(
					ValueToValueRange(dfDummy, 'year', totalEnd - year_end))

				dfDummy['year'] = year_end
				dfDummy['value'] = 0
				dfAdd_out = dfAdd_out.append(
					ValueToValueRange(dfDummy, 'year', totalEnd - year_end))

			OutputToFile(dfRate_out, 'wand_checks/{}_mult'.format(self.name))
			OutputToFile(dfAdd_out, 'wand_checks/{}_add'.format(self.name))

			dfRate_out = dfRate_out.reset_index()
			dfAdd_out  = dfAdd_out.reset_index()
			self.rate_mult = builder.lookup.build_table(dfRate_out, 
												key_columns=['sex', 'strata'], 
												parameter_columns=['age','year'])
			self.rate_add = builder.lookup.build_table(dfAdd_out, 
												key_columns=['sex', 'strata'], 
												parameter_columns=['age','year'])

			if 'target' in configuration:
				builder.value.register_value_modifier(configuration.target, self.rate_adjustment)
			else:
				builder.value.register_value_modifier(self.name, self.rate_adjustment)


	def rate_adjustment(self, index, rates):
		if self.doSet:
			return rates * self.rate_mult(index) + self.rate_add(index)
		return rates * self.rate_mult(index)


class ModifyAcuteDiseaseYLD:

	def __init__(self, name):
		self._name = name
		
	@property
	def name(self):
		return self._name

	def setup(self, builder):
		self.config = builder.configuration
		self.scale = self.config.intervention[self.name].yld_scale
		if self.scale < 0:
			raise ValueError(f'Invalid YLD scale: {self.scale}')
		builder.value.register_value_modifier(
			f'{self.name}_intervention.yld_rate',
			self.disability_adjustment)

	def disability_adjustment(self, index, rates):
		return rates * self.scale


class ModifyAcuteDiseaseMortality:

	def __init__(self, name):
		self._name = name
	
	@property
	def name(self):
		return self._name

	def setup(self, builder):
		self.config = builder.configuration
		self.scale = self.config.intervention[self.name].mortality_scale
		if self.scale < 0:
			raise ValueError(f'Invalid mortality scale: {self.scale}')
		builder.value.register_value_modifier(
			f'{self.name}_intervention.excess_mortality',
			self.mortality_adjustment)

	def mortality_adjustment(self, index, rates):
		return rates * self.scale
