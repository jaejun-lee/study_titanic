import importlib as imt
import DecisionTree
imt.reload(DecisionTree)

from DecisionTree import DecisionTree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from collections import Counter

#%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


class RandomForest(object):
    '''A Random Forest class'''

    def __init__(self, num_trees, num_features):
        '''
           num_trees:  number of trees to create in the forest:
        num_features:  the number of features to consider when choosing the
                           best split for each node of the decision trees
        '''
        self.num_trees = num_trees
        self.num_features = num_features
        self.forest = None

    def fit(self, X, y):
        '''
        X:  two dimensional numpy array representing feature matrix
                for test data
        y:  numpy array representing labels for test data
        '''
        self.forest = self.build_forest(X, y, self.num_trees,
                                        self.num_features)

    def build_forest(self, X, y, num_trees, num_features):
        '''
        Return a list of num_trees DecisionTrees built using bootstrap samples
        and only considering num_features features at each branch.
        '''
        lst_tree = []
        lst_index = list(range(0, len(y)))

        sample_size = len(y)
        for num in range(num_trees):
            sample_index = np.random.choice(lst_index, size=sample_size)
            sample_X = X[sample_index]
            sample_y = y[sample_index]
            dt = DecisionTree(num_features=num_features)
            dt.fit(sample_X, sample_y)
            lst_tree.append(dt)
        
        return lst_tree
            

    def predict(self, X):
        '''
        Return a numpy array of the labels predicted for the given test data.
        '''
        y_hats = []
        for tree in self.forest:
            y_hats.append(tree.predict(X))
        
        y_hats = np.array(y_hats).T
        y_hat = [np.unique(y_row)[0] for y_row in y_hats]
        
        return np.array([Counter(row).most_common(1)[0][0] for row in y_hats])
        #return y_hat

            

        


    def score(self, X, y):
        '''
        Return the accuracy of the Random Forest for the given test data and
        labels.
        '''
        
        y_hat = self.predict(X)
        TF = np.equal(y, y_hat)
        return np.sum(TF)/len(TF)


if __name__=="__main__":
    # df = pd.read_csv('../data/playgolf.csv')

    # df_X = df.copy()
    # df_y = df_X.pop("Result")
    # df_y = df_y.map({"Don't Play": 0, "Play": 1}
    

    

    # rf = RandomForest(num_trees=10, num_features=2)
    # rf.fit(X_train, y_train)
    # y_predict = rf.predict(X_test)
    # print("score:", rf.score(X_test, y_test))

    # dt = DecisionTree()
    # dt.fit(X_train, y_train)
    # predicted_y = dt.predict(X_test)
    # df = pd.read_csv('../data/playgolf.csv')
    # y = df.pop('Result').values
    # X = df.values
    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    # rf = RandomForest(num_trees=10, num_features=2)
    # rf.fit(X_train, y_train)
    # y_predict = rf.predict(X_test)

    # df = pd.read_csv('../data/playgolf.csv')
    # y = df.pop('Result').values
    # X = df.values
    # X_train, X_test, y_train, y_test = train_test_split(X, y)
    # rf = RandomForest(num_trees=10, num_features=2)
    # rf.fit(X_train, y_train)
    # y_predict = rf.predict(X_test)
    # print(rf.score(X_train, y_train))
    # print(rf.score(X_test, y_test))


    #rfc = RandomForestClassifier(n_estimators=10)
    #rfc.fit(X_train, y_train)
    #y_predict = rfc.predict(X_test)
    #print(rfc.score(X_train, y_train))
    #print(rfc.score(X_test, y_test))
    
    df = pd.read_csv('../data/congressional_voting.csv', header=None)
    y = df.pop(0).values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    rf = RandomForest(num_trees=10, num_features=5)
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)
    print(rf.score(X_train, y_train))
    print(rf.score(X_test, y_test))


    dt = DecisionTree()
    dt.fit(X_train, y_train)
    y_predict = dt.predict(X_test)
    print(dt.score(X_train, y_train))
    print(dt.score(X_test, y_test))

    # rfc = RandomForestClassifier(n_estimators=10)
    # rfc.fit(X_train, y_train)
    # y_predict = rfc.predict(X_test)
    # print(rfc.score(X_train, y_train))
    # print(rfc.score(X_test, y_test))

    # scores = []
    # numoftrees = list(range(10, 100, 10))
    # for num in numoftrees:
    #     rf = RandomForest(num_trees=10, num_features=2)
    #     rf.fit(X_train, y_train)
    #     scores.append(rf.score(X_test, y_test))
    
    # plt.plot(numoftrees, scores)







    