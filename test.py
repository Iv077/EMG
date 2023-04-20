#test

import csv
import time
import myo

class EmgCollector(myo.DeviceListener):
  """
  Collects EMG data in a list with *n* maximum number of elements.
  """

  def __init__(self, n):
    self.n = n
    self.emg_data = []

  def get_emg_data(self):
    return self.emg_data

  # myo.DeviceListener

  def on_connected(self, event):
    event.device.stream_emg(True)

  def on_emg(self, event):
    self.emg_data.append(event.emg)

def main():
  myo.init()
  hub = myo.Hub()
  listener = EmgCollector(2772)
  with hub.run_in_background(listener.on_event):
    time.sleep(2)
  emg_data = listener.get_emg_data()
  with open('emg_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(emg_data)

if __name__ == '__main__':
  main()








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