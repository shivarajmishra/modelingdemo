from os import path
import pandas as pd
import numpy as np
import math


# Returns cs + cscv = total smoking prevalence for input age/sex/stratum
# Note: These are STARTING prevalence values for the population
def get_total_smoking_prevalence(dataDir, age, sex, stratum):
	df = pd.read_csv(f"{dataDir}/circuit/prevalence/prevalence.csv")
	df = df.loc[
		(df["agecategory"] == age) &
		(df["sex"] == sex) &
		(df["strata"] == stratum)
		]
	return (df["cs"] + df["cscv"]).values[0]

# Reads dataFilePre_{...}_22.csv, and returns the 20-year circuit-flows for input age/sex/stratum.
# 20-year circuit-flows are estimated by:
# (1) 	pulling out the initial (starting) total smoking prevalence for an input age/sex/stratum;
# (2) 	finding which 'year' along a 22y/o's (from same sex and stratum) circuit-flows gives total smoking prevalence 
# 		closest to the initial value in (1);
# (3) 	pulling out the flows for the next 20 years after the starting year found in (2).
# This is a hacky way to approximate smoking circuit-flow-cycle of a non-22-y/o, by selecting a
# representative starting point somewhere along the 22-y/o circuit-flow timeline.
def get_flows_cycle(dataDir, age, sex, stratum):
	if age < 42:
		print(f"- Age file: 22")
		path_to_flows = f"{dataDir}/circuit/pre_process/input_data/{stratum}_{sex}_22.csv"
	elif 42 <= age < 62:
		print(f"- Age file: 42")
		path_to_flows = f"{dataDir}/circuit/pre_process/input_data/{stratum}_{sex}_42.csv"
	else:
		print(f"- Age file: 62")
		path_to_flows = f"{dataDir}/circuit/pre_process/input_data/{stratum}_{sex}_62.csv"
	
	df = pd.read_csv(path_to_flows)

	if age < 20:
		starting_year_idx = 0
	else:
		total_smoking_prev = get_total_smoking_prevalence(
			dataDir,
			age=math.floor(age/5)*5, 
			sex=sex, 
			stratum=stratum
		)

		# print(df.head(50))
		starting_year_idx = df.iloc[(df["totalSmoke"]-total_smoking_prev).abs().argsort()[:1]].index[0]

		matched_smoking_prev = df.iloc[(df["totalSmoke"]-total_smoking_prev).abs().argsort()[:1]]["totalSmoke"].values[0]
		print(f"- Initial smoking prevalence: {round(total_smoking_prev,4)},\n- Matched value: {round(matched_smoking_prev,4)},\n- Difference: {round(matched_smoking_prev - total_smoking_prev,4)},\n- Year proxy: {starting_year_idx}")
		# print(combined_df.iloc[(combined_df["totalSmoke"]-total_smoking_prev).abs().argsort()].head(150))

	flows_cycle_df = df[starting_year_idx:(starting_year_idx + 21)][["DU-->FSFV","DU-->FSCV","CS-->FSCV","CS-->FSFV","DU-->CS","CS-->DU"]].reset_index(drop=True)
	# print(flows_cycle_df.head(20))

	return flows_cycle_df

# Populates dataFileTemplate.csv using flows extracted from method described above, outputs dataFile.csv
def populate_data_file(dataDir, ages, sexes, strata):
	data_file_df = pd.read_csv(f"{dataDir}/circuit/pre_process/input_data/dataFileTemplate.csv")

	for age in ages:
		for sex in sexes:
			for stratum in strata:
				print(
					"-----------------------------------------------------\n" +
					f"Processing age: {age}, {sex}, {stratum} ...\n" +
					"-----------------------------------------------------" 
					)
				# Get 20-year flows for the age, sex, stratum combo
				flows_cycle_df = get_flows_cycle(dataDir, age, sex, stratum)

				# Loop through 20 years, figure out what age category the person lands in,
				# then set the flows in dataFile.csv at the appropriate age/year indexes
				for year in range(0, len(flows_cycle_df)):
					age_bracket = math.floor((age+year)/5)*5 +2
					current_year_flows = flows_cycle_df.loc[year]
	
					idx = data_file_df.index[
						(data_file_df["year"] == year) &
						(data_file_df["age"] == age_bracket) &
						(data_file_df["sex"] == sex) &
						(data_file_df["strata"] == stratum) 
					]

					for flow in ["DU-->FSFV","DU-->FSCV","CS-->FSCV","CS-->FSFV","DU-->CS","CS-->DU"]:
						data_file_df.at[idx,flow] = current_year_flows[flow]  



	data_file_df.set_index(["year", "age", "sex", "strata"],inplace=True)
	# data_file_df.to_csv("src/mslt/artifacts/data/circuit/pre_process/test.csv")

	return data_file_df


# Creates cs.csv and cscv.csv. Simply resorting dataFile.csv and renaming columns
def create_flow_output(dataDir, ages, sexes, strata):
	data_file_df = populate_data_file(
		dataDir,
		ages=ages,
		sexes=sexes,
		strata=strata
	)

	data_file_df = data_file_df.reset_index()
	data_file_df = data_file_df.fillna(method="ffill")
	data_file_df["year_start"] = data_file_df["year"]
	data_file_df["year_end"] = data_file_df["year"] + 1
	data_file_df.loc[(data_file_df["year_end"]==21), ["year_end"]] = 120
	data_file_df["age"] = data_file_df["age"] - 2

	cs_df = data_file_df[["year_start", "year_end", "age", "sex", "strata", "CS-->DU", "CS-->FSFV", "CS-->FSCV"]].copy()
	cscv_df = data_file_df[["year_start", "year_end", "age", "sex", "strata", "DU-->CS", "DU-->FSFV", "DU-->FSCV"]].copy()

	rename_dict = {
		"age":"agecategory",
		"CS-->DU":"cscv", 
		"CS-->FSFV":"qsqv_1", 
		"CS-->FSCV":"qscv_1",	
		"DU-->CS":"cs", 
		"DU-->FSFV":"qsqv_1", 
		"DU-->FSCV":"qscv_1"	
	}
	cs_df.rename(rename_dict, axis='columns', inplace = True)
	cscv_df.rename(rename_dict, axis='columns', inplace = True)

	cs_df = (cs_df.sort_values(["year_start", "sex", "strata", "agecategory"], ascending=True)
              .set_index(["year_start", "year_end", "agecategory", "sex", "strata"]))
			  
	cscv_df = (cscv_df.sort_values(["year_start", "sex", "strata", "agecategory"], ascending=True)
			.set_index(["year_start", "year_end", "agecategory", "sex", "strata"]))

	cs_df.to_csv(f"{dataDir}/circuit/flow/cs.csv")
	cscv_df.to_csv(f"{dataDir}/circuit/flow/cscv.csv")

def MatchAndDiagonalIndexFlows(dataDir):
	print(dataDir)
	create_flow_output(
		dataDir,
		ages=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105],
		# ages=[70],
		sexes=["female", "male"],
		strata=["maori", "non-maori"]
	)



