import pandas as pd
import numpy as np
import os


class FeatureEngineering:
    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.list_report = [list_dir for list_dir in os.listdir(dir_name)]


if __name__ == "__main__":
    # Set directory of ariba run output
    directory = "C:\\Users\\Wildan PC\\Documents\\GitHub\\tuberculosis-resistance-classification\\ariba_out\\"

    # Get a class of FeatureEngineering
    data = FeatureEngineering(directory)

    # Get list of feature into dataframe column
    # var_list is a set of all unique mtb gene_variant from ariba run output report.tsv
    # where the gene_variant is composite of cluster and known_var_change column.
    var_list = set()
    var_list.update(["!accession"])
    print(data.list_report)

    for report in data.list_report:
        df = pd.read_table(directory + report)

        # Get only the record that have feature "var_seq_type" == "p"
        df = df[df["var_seq_type"] == "p"]

        # Choose only specified column into new dataframe
        df_feat = df[["cluster", "var_type", "var_seq_type", "known_var_change"]]
        df_temp = df_feat[["cluster", "known_var_change"]].copy()

        # Combine column "cluster" and "known_var_change" into new column about gene_variant
        df_temp = df_temp.apply("_".join, axis=1)
        temp = df_temp.to_numpy().tolist()

        # Update the var_sets
        var_list.update(temp)

    # Sort the mtb unique gene variant
    feature_df = pd.DataFrame(var_list).sort_values(by=[0])

    # Transpose the df, so the row become the column, and delete the first row of record
    # So this process result is empty dataframe consist of unique mtb gene_variant as a column name
    feature_df = feature_df.T
    feature_df.columns = feature_df.iloc[0]
    feature_df = feature_df[1:]

    # Insert all record data into row of dataframe
    for idx, report in enumerate(data.list_report):
        # Get the accession number of the report.tsv
        accession_number = report.split('_')[3]
        df = pd.read_table(directory + report)

        # Get only the record that have feature "var_seq_type" == "p"
        df = df[df["var_seq_type"] == "p"]

        # Choose only specified column into new dataframe
        df_feat = df[["cluster", "var_type", "var_seq_type", "known_var_change"]]
        df_temp = df_feat[["cluster", "known_var_change"]].copy()

        # Combine column "cluster" and "known_var_change" into new column about gene_variant
        df_temp = df_temp.apply("_".join, axis=1)
        temp = df_temp.to_numpy().tolist()

        # If the specified gene_variant at j-column of the dataframe is available on current report.tsv
        # set cell value into 1
        for var in temp:
            feature_df.at[idx, var] = 1
        # Insert report.tsv accession number into dataframe
        feature_df.at[idx, "!accession"] = accession_number

    # Fill NaN value into 0
    feature_df = feature_df.fillna(0)
    print(feature_df)

    feature_df.to_csv('amr_datasets.csv', index=False)
