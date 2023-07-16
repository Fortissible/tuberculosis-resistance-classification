import pandas as pd
import numpy as np
import os

# GET CLASS DF WHICH IS DF ONLY DATA ID AND IT'S LABELS
if __name__ == "__main__":

    # Get CSV of All class according to the data ID
    classes_df = pd.read_csv("alldata_sheet16_classes.csv")

    # Get CSV from FeatureEngineering.py process (one without label, just AMR only)
    amr_datasets = pd.read_csv("amr_datasets.csv")

    class_list = []
    for idx in range(len(amr_datasets)):
        res_list = []
        # Get Data ID (in this case is Accession number
        acc_num = amr_datasets.loc[idx,"!accession"]

        # Get Label for specific Data ID, here is 4 Label for a Accession number
        lineage = classes_df.loc[classes_df["!accession"] == acc_num, "lineage"]
        phen_inh = classes_df.loc[classes_df["!accession"] == acc_num, "phen_inh"]
        phen_rif = classes_df.loc[classes_df["!accession"] == acc_num, "phen_rif"]
        phen_emb = classes_df.loc[classes_df["!accession"] == acc_num, "phen_emb"]
        phen_pza = classes_df.loc[classes_df["!accession"] == acc_num, "phen_pza"]

        # This block code for append the result to the new DF
        res_list.append(acc_num)
        res_list.append(lineage.values[0])
        res_list.append(phen_inh.values[0])
        res_list.append(phen_rif.values[0])
        res_list.append(phen_emb.values[0])
        res_list.append(phen_pza.values[0])

        class_list.append(res_list)

    # Create amr_datasets_all_class_bin.csv from list
    class_df = pd.DataFrame(class_list,
                 columns=["!accession", 'lineage', 'phen_inh', 'phen_rif', "phen_emb", "phen_pza"])
    class_df = class_df.drop('!accession', axis=1)
    dataset_done = pd.concat([amr_datasets, class_df], axis=1, join='inner')
    dataset_done.to_csv('amr_datasets_all_class_bin.csv', index=False)