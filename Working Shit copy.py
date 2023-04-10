
#-------------  VERSION 1 ---------------#
#----   Prefered option since looks cleaner ----#

import pandas as pd     #   library to be able to access each of the csv files
import glob             #   library to be able to access the path 
import numpy as np      #   library that sumplifies mathematical operations
import matplotlib.pyplot as plt
import joblib


path = "Vaibhav/"           
all_files = glob.glob(path + "*.csv")


li = []
for filename in all_files:
    flat_list = []
    df =np.array((pd.read_csv(filename, index_col=1, delimiter=',', usecols=range(1,9))))#, skiprows=1
    for sublist in df:
        for item in sublist:
            flat_list.append(item)
    li.append(flat_list)


tar = ["Switch", "Freeze", "On/Off", "Forwards", "Backwards", "Left", "Right", "Up", "Down"]
n = 10
target = np.repeat(tar, n) # reperat each element n times
signal = np.array(li)



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]


X = signal
y = target

#X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(name, score)

# for i in range(len(y_test)):            # Helps to visualize the how are compared the cards and to see posible errors
#     print(y_test[i], y_test[i])
    

# filename = 'EMG_Classifier.sav'
# joblib.dump(clf, filename)   