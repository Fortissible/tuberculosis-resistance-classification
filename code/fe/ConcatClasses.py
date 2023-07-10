import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    classes_df = pd.read_csv("alldata_sheet16_classes.csv")
    amr_datasets = pd.read_csv("amr_datasets.csv")
    amr_datasets["phen_inh"] = 0
    amr_datasets["phen_rif"] = 0
    amr_datasets["phen_emb"] = 0
    amr_datasets["phen_pza"] = 0

    class_list = []
    for idx in range(len(amr_datasets)):
        res_list = []
        acc_num = amr_datasets.loc[idx,"!accession"]
        phen_inh = classes_df.loc[classes_df["!accession"] == acc_num, "phen_inh"]
        phen_rif = classes_df.loc[classes_df["!accession"] == acc_num, "phen_rif"]
        phen_emb = classes_df.loc[classes_df["!accession"] == acc_num, "phen_emb"]
        phen_pza = classes_df.loc[classes_df["!accession"] == acc_num, "phen_pza"]

        res_list.append(acc_num)
        res_list.append(phen_inh.values[0])
        res_list.append(phen_rif.values[0])
        res_list.append(phen_emb.values[0])
        res_list.append(phen_pza.values[0])

        class_list.append(res_list)

    class_df = pd.DataFrame(class_list,
                 columns=["!accession", 'phen_inh', 'phen_rif', "phen_emb", "phen_pza"])
    class_df.to_csv('class_df.csv', index=False)