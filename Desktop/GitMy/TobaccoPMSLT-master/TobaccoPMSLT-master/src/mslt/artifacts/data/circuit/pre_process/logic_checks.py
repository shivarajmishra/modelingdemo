import pandas as pd

# First produce cs.csv and cscv.csv with make_input_tables.py
# This function checks that:
# 1) there are no negative values in cs.csv or cscv.csv,
# 2) the all rows in cs.csv and cscv.csv sum to less than 1
def do_logic_checks():
	cs_df = pd.read_csv("src/mslt/artifacts/data/circuit/flow/cs.csv").set_index(["year_start", "year_end", "agecategory", "sex", "strata"])
	cscv_df = pd.read_csv("src/mslt/artifacts/data/circuit/flow/cscv.csv").set_index(["year_start", "year_end", "agecategory", "sex", "strata"])
	
	logic_test_passed = True

	# Check no negative values
	if (cs_df.values < 0).any() or (cscv_df.values < 0).any():
		logic_test_passed = False
		cs_neg_value_idx = list(cs_df[(cs_df < 0).all(1)].index)
		cscv_neg_value_idx = list(cscv_df[(cs_df < 0).all(1)].index)
		print(
			"Negative value(s) in dataframe(s).\n" +
			f"(cs.csv) Index of negative values: {cs_neg_value_idx}\n" +
			f"(cscv.csv) Index of negative values: {cscv_neg_value_idx}\n"
			)

	# Check rows don't sum to more than 1
	cs_df["row_sum"] = cs_df.sum(axis=1)
	cs_df_sum_greater_than_1 = cs_df["row_sum"].loc[cs_df["row_sum"] > 1.0001].sum()
	cscv_df["row_sum"] = cscv_df.sum(axis=1)
	cscv_df_sum_greater_than_1 = cscv_df["row_sum"].loc[cscv_df["row_sum"] > 1.0001].sum()

	if (cs_df_sum_greater_than_1 != 0) or (cscv_df_sum_greater_than_1 != 0):
		logic_test_passed = False
		cs_problematic_row_idx = list(cs_df["row_sum"].loc[cs_df["row_sum"] > 1.0001].index)
		cscv_probelmatic_row_idx = list(cscv_df["row_sum"].loc[cscv_df["row_sum"] > 1.0001].index)
		print(
			"Row(s) sum to more than 1 in dataframe(s).\n" +
			f"(cs.csv) Index of problematic rows: {cs_problematic_row_idx}\n" +
			f"(cscv.csv) Index of problematic rows: {cscv_probelmatic_row_idx}\n"
			)

	if logic_test_passed:
		print("Logic tests passed.")
	else:
		print("Logic tests failed.")



do_logic_checks()