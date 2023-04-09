
#-------------  VERSION 1 ---------------#
#----   Prefered option since looks cleaner ----#

import pandas as pd
import glob
import numpy as np
import joblib

path = "Angel/"
all_files = glob.glob(path + "*.csv")


li = []


for filename in all_files:
    df =pd.read_csv(filename, index_col=1, delimiter=',', skiprows=1)#, usecols=range(1,9))#, skiprows=1
    li.append(df)

target = ['Switch', 'Switch', 'Switch', 'Switch', 'Switch', 'Switch', 'Switch', 'Switch', 'Switch', 'Switch', 'Freeze', 'Freeze', 'Freeze', 'Freeze', 'Freeze', 'Freeze', 'Freeze', 'Freeze', 'Freeze', 'Freeze', 'On/Off', 'On/Off', 'On/Off', 'On/Off', 'On/Off', 'On/Off', 'On/Off', 'On/Off', 'On/Off', 'On/Off', 'Forwards', 'Forwards', 'Forwards', 'Forwards', 'Forwards', 'Forwards', 'Forwards', 'Forwards', 'Forwards', 'Forwards', 'Backwards', 'Backwards', 'Backwards', 'Backwards', 'Backwards', 'Backwards', 'Backwards', 'Backwards', 'Backwards', 'Backwards', 'Left', 'Left', 'Left', 'Left', 'Left', 'Left', 'Left', 'Left', 'Left', 'Left', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Right', 'Up', 'Up', 'Up', 'Up', 'Up', 'Up', 'Up', 'Up', 'Up', 'Up', 'Down', 'Down', 'Down', 'Down', 'Down', 'Down', 'Down', 'Down', 'Down', 'Down']
signal = (li)
print(signal)





# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# X = signal
# y = target
# clf = GaussianNB()        # Choosen clasifier 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
# clf.fit(X_train, y_train)
# score = clf.score(X_test, y_test)
# print(score*100, '%')

# for i in range(len(y_test)):            # Helps to visualize the how are compared the cards and to see posible errors
#     print(y_test[i], y_test[i])
    

# # filename = 'EMG_Classifier.sav'
# # joblib.dump(clf, filename)   