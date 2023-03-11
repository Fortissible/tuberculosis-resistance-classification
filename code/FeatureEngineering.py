import pandas as pd
import numpy as np
import os


class FeatureEngineering:
    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.dir_depth = [list_dir[1] for list_dir in os.walk(dir_name)]
        #self.data = pd.read_table(dir_name + "/report.tsv")
        #self.datas = [np.array(data["known_var_change"])]
