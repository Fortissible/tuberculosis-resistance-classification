from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from joblib import dump,load
import numpy as np
import pandas as pd
from code.DecisionTree import DecisionTree
from code.RFClassifier import RandomForest

def joblib_save_model(clf,file_name):
    dump(clf, file_name)

def joblib_load_mode(file_name):
    clf = load(file_name)
    return clf

def predit_using_saved_model(dataset,file_name):
    x, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]

    clf = joblib_load_mode(file_name)
    pred = clf.predict(x)

    clf_acc = accuracy(y, pred)
    print(clf_acc)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


def print_decision_rules(rf):
    for tree_idx, est in enumerate(rf.estimators_):
        tree = est.tree_
        assert tree.value.shape[1] == 1  # no support for multi-output

        print('TREE: {}'.format(tree_idx))

        iterator = enumerate(zip(tree.children_left, tree.children_right, tree.feature, tree.threshold, tree.value))
        for node_idx, data in iterator:
            left, right, feature, th, value = data

            # left: index of left child (if any)
            # right: index of right child (if any)
            # feature: index of the feature to check
            # th: the threshold to compare against
            # value: values associated with classes

            # for classifier, value is 0 except the index of the class to return
            class_idx = np.argmax(value[0])

            if left == -1 and right == -1:
                print('{} LEAF: return class={}'.format(node_idx, class_idx))
            else:
                print(
                    '{} NODE: if feature[{}] < {} then next={} else next={}'.format(node_idx, feature, th, left, right))


def scratch_dt(x_train, x_test, y_train, y_test):
    clf = DecisionTree(max_depth=100)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    acc = accuracy(y_test, pred)
    print("from scratch decision tree accuracy is :", acc)


def scikit_learn_dt(x_train, x_test, y_train, y_test, feat_names, save_model=False, model_name="scikit_learn_dt_model"):
    lib_clf = DecisionTreeClassifier(max_depth=100)
    lib_clf.fit(x_train, y_train)
    lib_pred = lib_clf.predict(x_test)

    lib_acc = accuracy(y_test, lib_pred)

    # generate the decision tree rules as text
    tree_rules = export_text(lib_clf, feature_names=feat_names)

    # print the decision tree rules
    print(tree_rules)

    print("from scikit_learn decision tree accuracy is :", lib_acc)

    if save_model:
        joblib_save_model(lib_clf,model_name+'.joblib')


def scratch_rf(x_train, x_test, y_train, y_test):
    clf = RandomForest(max_depth=100)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    acc = accuracy(y_test, pred)
    print("from scratch random forest accuracy is :", acc)


def scikit_learn_rf(x_train, x_test, y_train, y_test, feat_names, save_model=False, model_name="scikit_learn_rf_model"):
    lib_clf = RandomForestClassifier(max_depth=100)
    lib_clf.fit(x_train, y_train)
    lib_pred = lib_clf.predict(x_test)

    # print rf rules
    # print_decision_rules(lib_clf)

    # Get the feature importance scores
    importances = lib_clf.feature_importances_

    # Sort random forest feature importances
    feat_importances = list(zip(range(x_train.shape[1]), importances))
    sorted_feat_importances = sorted(feat_importances, key=lambda x: x[1], reverse=True)

    # Print the feature importance scores
    for idx, feat_importance in enumerate(sorted_feat_importances):
        if idx > 10:
            break
        else:
            print(f"Feature {feat_names[feat_importance[0]]}: {feat_importance[1]}")

    lib_acc = accuracy(y_test, lib_pred)
    print("\nfrom scikit_learn random forest accuracy is :", lib_acc)

    if save_model:
        joblib_save_model(lib_clf,model_name+'.joblib')


if __name__ == "__main__":
    print("res_type short name list : emb, inh, pza, rif")
    resistance_type = input("insert resistance type short name!\n")

    # Read dataset.csv into dataframe
    data = pd.read_csv(f'amr_datasets_r_{resistance_type}.csv')

    # Drop phen_r_inh feature (duplicate class in dataset.csv)
    data = data.drop(f'phen_r_{resistance_type}', axis=1)

    # Copy dataframe and drop the accession number feature on the new dataframe
    data_without_acc_num = data.copy()
    data_without_acc_num = data_without_acc_num.drop('!accession', axis=1)

    # Slice dataframe line_age column into new dataframe
    new_df = np.array(data_without_acc_num.loc[:, 'line_age'])

    # Do one hot encoding on line_age feature
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(new_df.reshape(-1, 1))
    line_age_onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out())

    # Combine one hot encoded line_age feature into old dataframe (dropped the not-coded line_age)
    data_without_acc_num = data_without_acc_num.drop('line_age', axis=1)
    phenotype_df = data_without_acc_num.loc[:, f'phen_{resistance_type}']
    data_without_acc_num = data_without_acc_num.drop(f'phen_{resistance_type}', axis=1)
    df = pd.concat([data_without_acc_num, line_age_onehot_df, phenotype_df], axis=1, join='inner')

    # Get feature names from dataframe
    feat_names = list(df.columns.values)[:-1]

    # Split data into train and test sets
    x, y = df.iloc[:, :-1], df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=2
    )

    print("------------------ DECISION TREE ------------------")
    scratch_dt(x_train, x_test, y_train, y_test)
    # scikit_learn_dt(x_train, x_test, y_train, y_test, feat_names, save_model=True, model_name=f"scikit_dt_{resistance_type}")

    print("\n\n------------------ RANDOM FOREST ------------------")
    scratch_rf(x_train, x_test, y_train, y_test)
    # scikit_learn_rf(x_train, x_test, y_train, y_test, feat_names, save_model=True, model_name=f"scikit_rf_{resistance_type}")
