import pandas as pd     
import glob             
import numpy as np      
import matplotlib.pyplot as plt
import joblib
import scipy as sp
from scipy.signal import filtfilt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
import math


path = "Iva - 90/"
all_files = glob.glob(path + "*.csv")

li = []
for filename in all_files:                                                                                                                                                                                                                                                                                                                                                                                              
    flat_list = []
    emg = pd.read_csv(filename, index_col=1, delimiter=',', usecols=range(1,9), nrows=390)
    
    # normalise cut-off frequencies to sampling frequency
    high_band = 20/(400/2)
    low_band = 50/(400/2)
    
    # create bandpass filter for EMG
    b1, a1 = sp.signal.butter(4, [high_band,low_band], btype='bandpass')
    
    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(b1, a1, emg, axis=0)    
    
    # process EMG signal: rectify
    emg_rectified = abs(emg_filtered)
    
    # create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass = 20/(400/2)
    b2, a2 = sp.signal.butter(4, low_pass, btype='lowpass')
    emg_envelope = sp.signal.filtfilt(b2, a2, emg_rectified, axis=0)

    for sublist in emg_envelope:
        for item in sublist:
            flat_list.append(item)
    li.append(flat_list)


tar = ["Switch", "Freeze", "On/Off", "Forw", "Back", "Left", "Right", "Up", "Down"]
n = 10
target = np.repeat(tar, n)
signal = np.array(li)

#####################   FEATURES    ########################

# Calculate mean absolute value (MAV) feature
mav = np.mean(signal)

# Calculate root mean square (RMS) feature
rms = math.sqrt(np.mean(np.square(signal)))

# Calculate variance of absolute values (VAR) feature
var = np.var(signal)

# Calculate waveform length (WL) feature
wl = np.sum(np.abs(np.diff(signal)))

# Calculate zero crossing (ZC) feature
zc = np.sum(np.abs(np.diff(np.sign(signal)))) / (2 * signal.size)

# Calculate the mean absolute value slope (MAVS) feature
mav_slope = np.mean(np.abs(np.diff(signal)))

# Calculate the number of crossings of mean (NCM) feature
ncm = np.sum(np.abs(np.diff(np.sign(signal - mav / 2)))) / (2 * len(signal))

# Calculate the difference absolute mean value (DAMV) feature
damv = np.median(np.abs(signal - mav))

# Calculate the simple Square Integral (ssi) feature
ssi = np.sum(signal**2,axis=0)

# Calculate the absolute differential signal (ADS) feature
ads = np.sum(np.abs(np.diff(signal,axis=0)),axis=0)



X = signal/mav
y = target
clf = SVC(kernel="linear", C=0.025)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score*100, '%')
# unique_targets = np.unique(y_test)

# y_pred = clf.predict(X_test)
# # unique_predictions = np.unique(y_pred)

# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(5.5, 5.5))
# percentages = cm / cm.sum(axis=1)[:, np.newaxis] * 100  # Fix here
# sns.heatmap(percentages, annot=True, cmap="Greens", fmt='.1f', xticklabels=str(tar), yticklabels=str(tar),vmin=0, vmax=100, cbar=False)
# plt.title('Subject 4 / Random Forest')
# plt.ylabel('Expected')
# plt.xlabel('Predicted')
# ax = plt.gca()
# ax.set_xticklabels(tar, rotation=90, ha='center')
# ax.set_yticklabels(tar, rotation=0, va='center')
# plt.show()

file = 'EMG_Classifier1.sav'
joblib.dump(clf, file)
