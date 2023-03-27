from code.fe.FeatureEngineering import FeatureEngineering

if __name__ == "__main__":
    dir_name = "C:/Users/Wildan PC/AppData/Local/Packages/CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc"\
                        "/LocalState/rootfs/home/wildanfajri1alfarabi/Kuliah"
    dataset = FeatureEngineering(dir_name)
    print(dataset.dir_depth[0])