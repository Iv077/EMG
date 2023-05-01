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
import scipy.signal as sig
import statistics as stats
import math



path = "Iva - 90/"
all_files = glob.glob(path + "*.csv")

li = []
for filename in all_files:
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

    #emg = np.array(emg)
    flat_list = [item for sublist in emg_envelope for item in sublist]
    li.append(flat_list)


raw = np.array(li)

#####################   FEATURES    ########################

# Calculate mean absolute value (MAV) feature
mav = np.mean(raw)

# Calculate root mean square (RMS) feature
rms = math.sqrt(np.mean(np.square(raw)))

# Calculate variance of absolute values (VAR) feature
var = np.var(raw)

# Calculate waveform length (WL) feature
wl = np.sum(np.abs(np.diff(raw)))

# Calculate zero crossing (ZC) feature
zc = np.sum(np.abs(np.diff(np.sign(raw)))) / (2 * raw.size)

# Calculate the mean absolute value slope (MAVS) feature
mav_slope = np.mean(np.abs(np.diff(raw)))

# Calculate the number of crossings of mean (NCM) feature
ncm = np.sum(np.abs(np.diff(np.sign(raw - mav / 2)))) / (2 * len(raw))

# Calculate the difference absolute mean value (DAMV) feature
damv = np.median(np.abs(raw - mav))

# Calculate the simple Square Integral (ssi) feature
ssi = np.sum(raw**2,axis=0)

# Calculate the absolute differential signal (ADS) feature
ads = np.sum(np.abs(np.diff(raw,axis=0)),axis=0)

    
features_list = [mav, rms, var, wl, zc, mav_slope, ncm, damv, ssi, ads]

for feature in features_list:

    # Extract features using the selected feature extraction method
    #X = np.array([features(emg) for emg in raw])
    X = li/feature

    #tar = ["Horn", "Fist", "Victory", "Rotation", "Up", "Down", "Spread", "OK"]
    tar = ["Switch", "Freeze", "On/Off", "Forw", "Back", "Left", "Right", "Up", "Down"]

    from sklearn.decomposition import PCA

    # perform PCA on emg_envelope
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(li)

    n = 20
    target = np.repeat(tar, n)
    y = target
    clf = SVC(kernel="linear", C=0.025)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    clf.fit(X_train, y_train)

    # Evaluate classification performance
    score = clf.score(X_test, y_test)
    #print('Classification score with PCA:', score*100, '%')
    print('Classification score for', feature, ':', score*100, '%')



# file = 'EMG_Classifier1.sav'
# joblib.dump(clf, file)