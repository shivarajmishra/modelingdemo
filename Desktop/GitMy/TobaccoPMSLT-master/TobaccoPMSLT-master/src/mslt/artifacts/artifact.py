import datetime
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from vivarium.framework.artifact import hdf
from vivarium.framework.artifact import Artifact

from mslt.artifacts.population import Population
from mslt.artifacts.disease import Diseases
from mslt.artifacts.circuit import Circuit
from mslt.artifacts.uncertainty import Normal, Beta, LogNormal
from mslt.utilities import get_data_dir
import mslt.utilities as util
from mslt.artifacts.population import Population
from mslt.artifacts.circuit_flow_preprocess import MatchAndDiagonalIndexFlows

YEAR_START = 2021
RANDOM_SEED = 49430
WRITE_CSV = True
WRITE_DISEASES = True
PRE_PROCESS_PREVALENCE = True
DATA_DIR = 'data'

def output_csv_mkdir(data, path):
	"""
	Wrapper for pandas .to_csv() method to create directory for path if it
	doesn't already exist.
	"""
	output_path = Path('.').resolve() / 'artifacts' / (path + '.csv')
	out_folder = os.path.dirname(output_path)

	if not os.path.exists(out_folder):
		os.mkdir(out_folder)

	print(output_path)
	data.to_csv(output_path)


def check_for_bin_edges(df):
	"""
	Check that lower (inclusive) and upper (exclusive) bounds for year and age
	are defined as table columns.
	"""

	if 'age_start' in df.columns and 'year_start' in df.columns:
		return df
	else:
		raise ValueError('Table does not have bins')


def write_table(artifact, path, data):
	"""
	Write a data table to an artifact, after ensuring that it doesn't contain
	any NA values.

	:param artifact: The artifact object.
	:param path: The table path.
	:param data: The table data.
	"""
	logger = logging.getLogger(__name__)
	logger.info('{} Writing table {} to {}'.format(
		datetime.datetime.now().strftime("%H:%M:%S"), path, artifact.path))

	#Add age,sex,year etc columns to multi index
	col_index_filters = ['year','age','sex', 'strata', 'year_start','year_end','age_start','age_end']
	data.set_index([col_name for col_name in data.columns if col_name in col_index_filters], inplace =True)
	
	if WRITE_CSV:
	  output_csv_mkdir(data, path)

	if np.any(data.isna()):
		msg = 'NA values in table {} for {}'.format(path, artifact.path)
		raise ValueError(msg)

	artifact.write(path, data)


def assemble_artifacts(num_draws, output_path: Path, seed: int = RANDOM_SEED):
	"""
	Parameters
	----------
	num_draws
		The number of random draws to sample for each rate and quantity,
		for the uncertainty analysis.
	output_path
		The path to the artifact being assembled.
	seed
		The seed for the pseudo-random number generator used to generate the
		random samples.

	"""

	data_dir = get_data_dir(DATA_DIR)
	prng = np.random.RandomState(seed=seed)
	logger = logging.getLogger(__name__)

	if PRE_PROCESS_PREVALENCE:
		MatchAndDiagonalIndexFlows(data_dir) # BESPOKE
	max_age = 110

	util.SetStrataFile('{}/strata.csv'.format(data_dir))

	# Instantiate components for the non-Maori population.
	pop = Population(data_dir, YEAR_START)
	if WRITE_DISEASES:
		diseaseList = Diseases(data_dir, YEAR_START, pop.year_end)

	# Define data structures to record the samples from the unit interval that
	# are used to sample each rate/quantity, so that they can be correlated
	# across both populations.
	smp_yld = prng.random_sample(num_draws)
	smp_chronic_apc = {}
	smp_chronic_i = {}
	smp_chronic_r = {}
	smp_chronic_f = {}
	smp_chronic_yld = {}
	smp_chronic_prev = {}
	smp_acute_f = {}
	smp_acute_yld = {}
	smp_acute_exp = {}

	# Define the sampling distributions in terms of their family and their
	# *relative* standard deviation; they will be used to draw samples for
	# both populations.
	dist_yld = LogNormal(sd_pcnt=10)
	dist_chronic_apc = Normal(sd_pcnt=0.5)
	dist_chronic_i = Normal(sd_pcnt=5)
	dist_chronic_r = Normal(sd_pcnt=5)
	dist_chronic_f = Normal(sd_pcnt=5)
	dist_chronic_yld = Normal(sd_pcnt=10)
	dist_chronic_prev = Normal(sd_pcnt=5)
	dist_acute_f = Normal(sd_pcnt=10)
	dist_acute_yld = Normal(sd_pcnt=10)
	dist_acute_expenditure_rate = Normal(sd_pcnt=10)

	logger.info('{} Generating samples'.format(
		datetime.datetime.now().strftime("%H:%M:%S")))

	if WRITE_DISEASES:
		for name, disease_nm in diseaseList.chronic.items():
			# Draw samples for each rate/quantity for this disease.
			smp_chronic_apc[name] = prng.random_sample(num_draws)
			smp_chronic_i[name] = prng.random_sample(num_draws)
			smp_chronic_r[name] = prng.random_sample(num_draws)
			smp_chronic_f[name] = prng.random_sample(num_draws)
			smp_chronic_yld[name] = prng.random_sample(num_draws)
			smp_chronic_prev[name] = prng.random_sample(num_draws)

		for name, disease_nm in diseaseList.acute.items():
			# Draw samples for each rate/quantity for this disease.
			smp_acute_f[name] = prng.random_sample(num_draws)
			smp_acute_yld[name] = prng.random_sample(num_draws)
			smp_acute_exp[name] = prng.random_sample(num_draws)

	# Now write all of the required tables
	artifact_fmt = 'pmslt_artifact.hdf'
	artifact_file = output_path / artifact_fmt

	logger.info('{} Generating artifacts'.format(
		datetime.datetime.now().strftime("%H:%M:%S")))

	# Initialise each artifact file.
	for path in [artifact_file]:
		if path.exists():
			path.unlink()

	# Write the data tables to each artifact file.
	art_nm = Artifact(str(artifact_file))

	# Write the main population tables.
	logger.info('{} Writing population tables'.format(
		datetime.datetime.now().strftime("%H:%M:%S")))
	write_table(art_nm, 'population.structure',
				 pop.get_population())
	write_table(art_nm, 'cause.all_causes.disability_rate',
				 pop.sample_disability_rate_from(dist_yld, smp_yld))
	write_table(art_nm, 'cause.all_causes.mortality',
				 pop.get_mortality_rate())
	write_table(art_nm, 'cause.all_causes.expenditure_rate',
				 pop.get_expenditure_rate())
	write_table(art_nm, 'cause.all_causes.expenditure_rate_death',
				 pop.get_expenditure_rate_death())
	write_table(art_nm, 'cause.all_causes.income',
				 pop.get_income())
	write_table(art_nm, 'cause.all_causes.income_death',
				 pop.get_income_death())

	if WRITE_DISEASES:
		# Write the chronic disease tables.
		for name, disease_nm in diseaseList.chronic.items():
			logger.info('{} Writing tables for {}'.format(
				datetime.datetime.now().strftime("%H:%M:%S"), name))

			write_table(art_nm, 'chronic_disease.{}.incidence'.format(name),
						disease_nm.sample_from('i',
							dist_chronic_i, dist_chronic_apc,
							smp_chronic_i[name], smp_chronic_apc[name]))
			write_table(art_nm, 'chronic_disease.{}.remission'.format(name),
						disease_nm.sample_from('r',
							dist_chronic_r, dist_chronic_apc,
							smp_chronic_r[name], smp_chronic_apc[name]))
			write_table(art_nm, 'chronic_disease.{}.mortality'.format(name),
						disease_nm.sample_from('f',
							dist_chronic_f, dist_chronic_apc,
							smp_chronic_f[name], smp_chronic_apc[name]))
			write_table(art_nm, 'chronic_disease.{}.morbidity'.format(name),
						disease_nm.sample_from('DR',
							dist_chronic_yld, dist_chronic_apc,
							smp_chronic_yld[name], smp_chronic_apc[name]))
			write_table(art_nm, 'chronic_disease.{}.expenditure_rate'.format(name),
						disease_nm.sample_from('expenditure_rate',
							dist_chronic_yld, dist_chronic_apc,
							smp_chronic_yld[name], smp_chronic_apc[name]))
			write_table(art_nm, 'chronic_disease.{}.expenditure_rate_first'.format(name),
						disease_nm.sample_from('expenditure_rate_first',
							dist_chronic_yld, dist_chronic_apc,
							smp_chronic_yld[name], smp_chronic_apc[name]))
			write_table(art_nm, 'chronic_disease.{}.expenditure_rate_last'.format(name),
						disease_nm.sample_from('expenditure_rate_last',
							dist_chronic_yld, dist_chronic_apc,
							smp_chronic_yld[name], smp_chronic_apc[name]))
			write_table(art_nm, 'chronic_disease.{}.income'.format(name),
						disease_nm.sample_from('income',
							dist_chronic_yld, dist_chronic_apc,
							smp_chronic_yld[name], smp_chronic_apc[name]))
			write_table(art_nm, 'chronic_disease.{}.income_first'.format(name),
						disease_nm.sample_from('income_first',
							dist_chronic_yld, dist_chronic_apc,
							smp_chronic_yld[name], smp_chronic_apc[name]))
			write_table(art_nm, 'chronic_disease.{}.income_last'.format(name),
						disease_nm.sample_from('income_last',
							dist_chronic_yld, dist_chronic_apc,
							smp_chronic_yld[name], smp_chronic_apc[name]))
			write_table(art_nm, 'chronic_disease.{}.prevalence'.format(name),
						disease_nm.sample_prevalence_from(
							dist_chronic_prev, smp_chronic_prev[name]))

		# Write the acute disease tables.
		for name, disease_nm in diseaseList.acute.items():
			logger.info('{} Writing tables for {}'.format(
				datetime.datetime.now().strftime("%H:%M:%S"), name))

			write_table(art_nm, 'acute_disease.{}.mortality'.format(name),
						disease_nm.sample_from('excess_mortality',
							dist_acute_f, smp_acute_f[name]))
			write_table(art_nm, 'acute_disease.{}.morbidity'.format(name),
						disease_nm.sample_from('disability_rate',
							dist_acute_yld, smp_acute_yld[name]))
			write_table(art_nm, 'acute_disease.{}.expenditure_rate'.format(name),
						disease_nm.sample_from('expenditure_rate',
							dist_acute_expenditure_rate, smp_acute_exp[name]))
			write_table(art_nm, 'acute_disease.{}.income'.format(name),
						disease_nm.sample_from('income',
							dist_acute_expenditure_rate, smp_acute_exp[name]))
	
	# Write the compartment circuit tables.
	logger.info('{} Writing compartment circuit tables'.format(
		datetime.datetime.now().strftime("%H:%M:%S")))
	Circuit(art_nm, num_draws, data_dir, write_table, YEAR_START, pop.year_end, max_age, WRITE_DISEASES, prng)

	print(artifact_file)
