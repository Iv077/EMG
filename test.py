# import pandas as pd     #   library to be able to access each of the csv files
# import glob             #   library to be able to access the path
# import numpy as np      #   library that sumplifies mathematical operations
# import matplotlib.pyplot as plt
# import joblib
# import scipy as sp
# from scipy.signal import filtfilt


# path = "Gen/"
# all_files = glob.glob(path + "*.csv")


# li = []
# for filename in all_files:                                                                                                                                                                                                                                                                                                                                                                                              
#     flat_list = []
#     emg = pd.read_csv(filename, index_col=1, delimiter=',', usecols=range(1,9), nrows=390)
#     emg_correctmean = emg - np.mean(emg, axis=0)
#     low_pass=40 # low: low-pass cut off frequency
#     sfreq=1000 # sfreq: sampling frequency
#     high_band=40
#     low_band=450
#     # emg: EMG data
#     # high: high-pass cut off frequency
    
#     # normalise cut-off frequencies to sampling frequency
#     high_band = high_band/(sfreq/2)
#     low_band = low_band/(sfreq/2)

#     # create bandpass filter for EMG
#     b1, a1 = sp.signal.butter(4, [high_band,low_band], btype='bandpass')

#     # process EMG signal: filter EMG
#     emg_filtered = sp.signal.filtfilt(b1, a1, emg, axis=0)

#     # process EMG signal: rectify
#     emg_rectified = abs(emg_filtered)

#     # create lowpass filter and apply to rectified signal to get EMG envelope
#     low_pass = low_pass/(sfreq/2)
#     b2, a2 = sp.signal.butter(4, low_pass, btype='lowpass')
#     emg_envelope = np.array(sp.signal.filtfilt(b2, a2, emg_rectified, axis=0))


#     for sublist in emg_envelope:
#         for item in sublist:
#             flat_list.append(item)
#     li.append(flat_list)
# #print(li)



# tar = ["Switch", "Freeze", "On/Off", "Forwards", "Backwards", "Left", "Right", "Up", "Down"]
# #tar = ["Left", "Right", "Forwards", "Switch", "Station"]
# n = 20
# target = np.repeat(tar, n) # reperat each element n times
# signal = np.array(li)



# import numpy as np
# from sklearn.model_selection import train_test_split

# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.linear_model import SGDClassifier

# names = ["Nearest_Neighbors", "Linear_SVM", "Polynomial_SVM", "RBF_SVM", "Gaussian_Process",
#          "Gradient_Boosting", "Decision_Tree", "Extra_Trees", "Random_Forest", "Neural_Net", "AdaBoost",
#          "Naive_Bayes", "QDA", "SGD"]

# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(kernel="poly", degree=3, C=0.025),
#     SVC(kernel="rbf", C=1, gamma=2),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     GradientBoostingClassifier(n_estimators=100, learning_rate=1.0),
#     DecisionTreeClassifier(max_depth=5),
#     ExtraTreesClassifier(n_estimators=10, min_samples_split=2),
#     RandomForestClassifier(max_depth=5, n_estimators=100),
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(n_estimators=100),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis(),
#     SGDClassifier(loss="hinge", penalty="l2")]


# X = signal
# y = target

# #X = StandardScaler().fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
# scores = []

# for name, clf in zip(names, classifiers):
#     clf.fit(X_train, y_train)
#     score = clf.score(X_test, y_test)
#     scores.append(score)
#     print(scores)






import pandas as pd     #   library to be able to access each of the csv files
import glob             #   library to be able to access the path 
import numpy as np      #   library that sumplifies mathematical operations
import joblib
import matplotlib.pyplot as plt

path = "Gen/"
all_files = glob.glob(path + "*.csv")


li = []


for filename in all_files:
    flat_list = []
    df =np.array((pd.read_csv(filename, index_col=1, delimiter=',', usecols=range(1,9), nrows=390)))#, skiprows=1
    for sublist in df:
        for item in sublist:
            flat_list.append(item)
    li.append(flat_list)


tar = ["Switch", "Freeze", "On/Off", "Forwards", "Backwards", "Left", "Right", "Up", "Down"]
#tar = ["Left", "Right", "Forwards", "Switch", "Station"]
n = 20
target = np.repeat(tar, n) # reperat each element n times
signal = np.array(li)



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier

names = ["Nearest_Neighbors", "Linear_SVM", "Polynomial_SVM", "RBF_SVM", "Gaussian_Process",
         "Gradient_Boosting", "Decision_Tree", "Extra_Trees", "Random_Forest", "Neural_Net", "AdaBoost",
         "Naive_Bayes", "QDA", "SGD"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(kernel="poly", degree=3, C=0.025),
    SVC(kernel="rbf", C=1, gamma=2),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0),
    DecisionTreeClassifier(max_depth=5),
    ExtraTreesClassifier(n_estimators=10, min_samples_split=2),
    RandomForestClassifier(max_depth=5, n_estimators=100),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(n_estimators=100),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    SGDClassifier(loss="hinge", penalty="l2")]


X = signal
y = target

#X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)

    print("Accuracy:", score*100, '%')
    cm = confusion_matrix(y_test, y_pred, labels=[tar])
    print("Confusion Matrix:\n", cm)

    import seaborn as sns
    sns.heatmap(cm)
    plt.show()





#test

# import csv
# import time
# import myo

# class EmgCollector(myo.DeviceListener):
#   """
#   Collects EMG data in a list with *n* maximum number of elements.
#   """

#   def __init__(self, n):
#     self.n = n
#     self.emg_data = []

#   def get_emg_data(self):
#     return self.emg_data

#   # myo.DeviceListener

#   def on_connected(self, event):
#     event.device.stream_emg(True)

#   def on_emg(self, event):
#     self.emg_data.append(event.emg)

# def main():
#   myo.init()
#   hub = myo.Hub()
#   listener = EmgCollector(2772)
#   with hub.run_in_background(listener.on_event):
#     time.sleep(2)
#   emg_data = listener.get_emg_data()
#   with open('emg_data.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(emg_data)

# if __name__ == '__main__':
#   main()



# with open('myfile.txt', 'r') as f:
#     contents = f.read()
#     print(contents)




########################################################################
      # HOST = "192.168.8.50"  # IP address of turtlebot robot
      # PORT = 3020  # port number for socket communication
      # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      # sock.connect((HOST, PORT))
      # if classi == ["Switch"]:
      #   message = "Switch".encode()
      #   sock.send(message)
      #   print("Switch") 

      # if classi == ["Freeze"]:
      #   message = "Freeze".encode()
      #   sock.send(message)
      #   print("Freeze")

      # if classi == ["On/Off"]:
      #   message = "On/Off".encode()
      #   sock.send(message)
      #   print("On/Off")

      # if classi == ["Forwards"]:
      #   message = "Forward".encode()
      #   sock.send(message)
      #   print("Forwards")

      # if classi == ["Backwards"]:
      #   message = "Backward".encode()
      #   sock.send(message)
      #   print("Backards")

      # if classi == ["Left"]:
      #   message = "left".encode()
      #   sock.send(message)
      #   print("Left")
        
      # if classi == ["Right"]:
      #   message = "right".encode()
      #   sock.send(message)
      #   print("Right")

      # if classi == ["Up"]:
      #   message = "up".encode()
      #   sock.send(message)
      #   print("Up")
        
      # if classi == ["Down"]:
      #   message = "down".encode()
      #   sock.send(message)
      #   print("Down")

      # else:
      #   print("Give me an order")




      #####################################################################################################


      #---------------Just to check -------------------#


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

# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1, max_iter=1000),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]

# names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
#          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes", "QDA"]


# X = signal
# y = target

# #X = StandardScaler().fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# for name, clf in zip(names, classifiers):
#     clf.fit(X_train, y_train)
#     score = clf.score(X_test, y_test)
#     print(name, score)