import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import score_model
import roc
from roc import *
import importlib as imt
imt.reload(score_model)
imt.reload(roc)

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split


from statsmodels.tools import add_constant
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import accuracy_score, precision_score, recall_score

from RandomForest import RandomForest


def analyze_unique_null_values(df):
    
    #Unique and Null Analysis
    df_report = pd.DataFrame.from_records([(c[0], c[1], df[c[1]].nunique(), df[c[1]].isnull().sum()) 
    for c in enumerate(df.columns)], index='index', columns=['index', 'name', '#ofUnique', '#ofNull'])
    print(df_report)

def design_matrix(df):

    df = df.copy()
    #Sex -> Female or not
    df['female'] = df.Sex.map({'male': 0, 'female': 1})
    del df['Sex']
    
    #Cabin -> has_Cabin
    df['has_Cabin'] = df.Cabin.notna().astype(int)
    
    del df['Cabin']

    #Age -> if null, fill to Age.mean() !!! impudiate candidate
    df.fillna(value={"Age": 29.70}, inplace=True)

    #Embarked -> hotcode
    # df_hotcode_Embarked = pd.get_dummies(df_train["Embarked"])
    # df_train = pd.concat([df_train, df_hotcode_Embakred], axis=1)
    # del df_train['Embarked']
    df = pd.get_dummies(df, columns=["Embarked"])

    del df["Name"]
    del df["Ticket"]

    #
    #del_low_pvalue(df)

    return df

def plot_roc_curve(df_X, df_y):

    X = df_X.to_numpy()
    y = df_y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    model = LogisticRegression(solver="lbfgs")
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    tpr, fpr, thresholds = roc_curve(probabilities, y_test)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(fpr, tpr)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity, Recall)")
    ax.set_title("ROC Plot of Titanic Data")

    return tpr, fpr, thresholds

def analyze_statsmodel(df_X, df_y):
    X = df_X.to_numpy()
    X_const = add_constant(X, prepend=True)
    y = df_y.to_numpy()

    logit_model = Logit(y, X_const).fit()
    print(logit_model.summary())

def del_low_pvalue(df):
    del df["Parch"] 
    del df["Fare"]
    del df["Embarked_C"]
    del df["Embarked_Q"] 
    del df["Embarked_S"]
    return df

def cv(df_X, df_y, k=10):
    
    kfold = KFold(n_splits=k)
    
    accuracies = []
    precisions = []
    recalls = []

    X = df_X.to_numpy()
    y = df_y.to_numpy()
    #X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    for train_index, test_index in kfold.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        model = LogisticRegression(solver="lbfgs")
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        # TP = np.multiply(y_test, y_hat).sum()
        # FP = np.multiply(np.subtract(1, y_test), y_hat).sum()
        # TN = np.multiply(np.subtract(1, y_test), np.subtract(1, y_hat)).sum()
        # FN = np.multiply(y_test, np.subtract(1, y_hat)).sum()
        # P = np.sum(y_test)
        # N = len(y_hat) - P

        accuracies.append(accuracy_score(y_test, y_hat))
        precisions.append(precision_score(y_test, y_hat))
        recalls.append(recall_score(y_test, y_hat))
    
    return np.mean(accuracies), np.mean(precisions), np.mean(recalls)

def cv_threshold(df_X, df_y, k=10, threshold=0.5):
    
    kfold = KFold(n_splits=k)
    
    accuracies = []
    precisions = []
    recalls = []

    X = df_X.to_numpy()
    y = df_y.to_numpy()
    #X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    for train_index, test_index in kfold.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        model = LogisticRegression(solver="lbfgs")
        model.fit(X_train, y_train)
        #y_hat = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        y_hat = probabilities >= threshold
        y_hat = y_hat.astype(int)

        # TP = np.multiply(y_test, y_hat).sum()
        # FP = np.multiply(np.subtract(1, y_test), y_hat).sum()
        # TN = np.multiply(np.subtract(1, y_test), np.subtract(1, y_hat)).sum()
        # FN = np.multiply(y_test, np.subtract(1, y_hat)).sum()
        # P = np.sum(y_test)
        # N = len(y_hat) - P

        accuracies.append(accuracy_score(y_test, y_hat))
        precisions.append(precision_score(y_test, y_hat))
        recalls.append(recall_score(y_test, y_hat))
    
    return np.mean(accuracies), np.mean(precisions), np.mean(recalls)

def analyze_RF(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    arr = list(range(50, 150, 10))
    scores = []
    for e in arr:
        model = RandomForest(num_trees=e, num_features=3)
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    
    fig, ax = plt.subplots()
    ax.plot(arr, scores)

    return scores, arr


def submit_prediction(y_prediction, df_test_X):
    df_test_X["Survived"]=y_prediction.tolist()
    df_survived = df_test_X["Survived"]
    df_survived = df_survived.reset_index()
    df_survived.to_csv("../data/submit.csv", columns=["PassengerId", "Survived"], index=False)


if __name__=='__main__':

    df_train = pd.read_csv("../data/train.csv", index_col="PassengerId")
    df_test = pd.read_csv("../data/test.csv", index_col="PassengerId")

    #analyze_unique_null_values(df_train)

    df_train_X = design_matrix(df_train)
    df_train_y = df_train_X.pop("Survived")

    #holdout X
    df_test_X = design_matrix(df_test)

    #good crosstab example. Survived, Pclass
    # survived = pd.crosstab(df_train["Survived"], df_train['Pclass'], rownames=['Survived'])
    # (survived / survived.apply(sum)).plot(kind='bar', figsize=(12, 6))

    #eda
    # df_train.hist()
    # df_train.Survived.value_counts() / len(df_train)

    #draw ROC Curve
    #plot_roc_curve(df_train_X, df_train_y)

    #statsmodel
    #analyze_statsmodel(df_train_X, df_train_y)
    # Parch, Fare, Embarked_* are p > 0.05. could drop them.

    #cross validation for accuracy, precision, fallout

    #print(cv(df_train_X, df_train_y))
    #print(cv_threshold(df_train_X, df_train_y))


    #threshold test
    # accs = []
    # pres = []
    # falls = []
    # thresholds = np.linspace(0, 1, 101)

    # for threshold in thresholds:
    #     acc, pre, fall = cv_threshold(df_train_X, df_train_y, threshold=threshold)
    #     accs.append(acc)
    #     pres.append(pre)
    #     falls.append(fall)

    # fig, ax = plt.subplots(figsize=(12, 6))
    # #ax.plot(thresholds, accs)
    # ax.plot(pres, falls)
    # ax.set_xlabel("threshold")
    # ax.set_ylabel("pres")
    # ax.set_title("precision")

    #predict test_X with threshold 0.53
    X_train = df_train_X.to_numpy()
    y_train = df_train_y.to_numpy()
    X_test = df_test_X.to_numpy()
    #model = LogisticRegression()

    #scores, arr = analyze_RF(X_train, y_train)

    model = RandomForest(num_trees=120, num_features=3)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)

    submit_prediction(y_hat)


    





    













