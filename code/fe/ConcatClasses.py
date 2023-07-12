import pandas as pd
import numpy as np
import os

# GET CLASS DF WHICH IS DF ONLY DATA ID AND IT'S LABELS
if __name__ == "__main__":

    # Get CSV of All class according to the data ID
    classes_df = pd.read_csv("alldata_sheet16_classes.csv")

    # Get CSV from FeatureEngineering.py process (one without label, just AMR only)
    amr_datasets = pd.read_csv("amr_datasets.csv")

    # Create label column on DF
    amr_datasets["phen_inh"] = 0
    amr_datasets["phen_rif"] = 0
    amr_datasets["phen_emb"] = 0
    amr_datasets["phen_pza"] = 0

    class_list = []
    for idx in range(len(amr_datasets)):
        res_list = []
        # Get Data ID (in this case is Accession number
        acc_num = amr_datasets.loc[idx,"!accession"]

        # Get Label for specific Data ID, here is 4 Label for a Accession number
        phen_inh = classes_df.loc[classes_df["!accession"] == acc_num, "phen_inh"]
        phen_rif = classes_df.loc[classes_df["!accession"] == acc_num, "phen_rif"]
        phen_emb = classes_df.loc[classes_df["!accession"] == acc_num, "phen_emb"]
        phen_pza = classes_df.loc[classes_df["!accession"] == acc_num, "phen_pza"]

        # This block code for append the result to the new DF
        res_list.append(acc_num)
        res_list.append(phen_inh.values[0])
        res_list.append(phen_rif.values[0])
        res_list.append(phen_emb.values[0])
        res_list.append(phen_pza.values[0])

        class_list.append(res_list)

    # Create class_df.csv from list
    class_df = pd.DataFrame(class_list,
                 columns=["!accession", 'phen_inh', 'phen_rif', "phen_emb", "phen_pza"])
    class_df.to_csv('class_df.csv', index=False)

    # After this code, copy class_df.csv labels into csv from FeatureEngineering.py process