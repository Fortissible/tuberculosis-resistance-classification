import pandas as pd
import numpy as np
import os

class FeatureEngineering:
    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.list_report = [list_dir for list_dir in os.listdir(dir_name)]
        #self.data = pd.read_table(dir_name + "/report.tsv")
        #self.datas = [np.array(data["known_var_change"])]

if __name__ == "__main__":
    directory = "C:\\Users\\wilda\\OneDrive\\Documents\\GitHub\\tuberculosis-resistance-classification\\ariba_out\\"
    data = FeatureEngineering(directory)
    var_list = set()
    for report in data.list_report:
        df = pd.read_table(directory+report)
        df = df[df["var_seq_type"] == "p"]
        df_feat = df[["cluster","var_type","var_seq_type","known_var_change"]]
        df_temp = df_feat[["cluster", "known_var_change"]].copy()
        df_temp = df_temp.apply("_".join, axis=1)
        temp = df_temp.to_numpy().tolist()
        print(len(temp))
        var_list.update(temp)
    feature_df = pd.DataFrame(var_list).sort_values(by=[0])
    feature_df.columns = ["var_feat"]
    print(feature_df)