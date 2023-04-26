#classi test
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
    emg = np.array(emg)
    flat_list = [item for sublist in emg for item in sublist]
    li.append(flat_list)

raw = np.array(li)
low_pass=40 # low: low-pass cut off frequency
sfreq=1000 # sfreq: sampling frequency
high_band=40
low_band=450

# normalise cut-off frequencies to sampling frequency
high_band = high_band/(sfreq/2)
low_band = low_band/(sfreq/2)

# create bandpass filter for EMG
b1, a1 = sp.signal.butter(4, [high_band,low_band], btype='bandpass')

# process EMG signal: filter EMG
emg_filtered = np.array([sp.signal.filtfilt(b1, a1, x) for x in li])

# process EMG signal: rectify
rect_signal = abs(emg_filtered)

#create lowpass filter and apply to rectified signal to get EMG envelope
nlow_pass = low_pass/(sfreq/2)
b2, a2 = sp.signal.butter(4, nlow_pass, btype='lowpass')
emg_envelope = np.array(sp.signal.filtfilt(b2, a2, rect_signal, axis=0))


#####################   FEATURES    ########################
def features(emg_envelope):
    # Calculate mean absolute value (MAV) feature
    mav = np.mean(emg_envelope)

    # Calculate root mean square (RMS) feature
    rms = math.sqrt(np.mean(np.square(emg_envelope)))

    # Calculate variance of absolute values (VAR) feature
    var = np.var(emg_envelope)

    # Calculate waveform length (WL) feature
    wl = np.sum(np.abs(np.diff(emg_envelope)))

    # Calculate zero crossing (ZC) feature
    zc = np.sum(np.abs(np.diff(np.sign(emg_envelope)))) / (2 * emg_envelope.size)

    # Calculate the mean absolute value slope (MAVS) feature
    mav_slope = np.mean(np.abs(np.diff(emg_envelope)))

    # Calculate the number of crossings of mean (NCM) feature
    ncm = np.sum(np.abs(np.diff(np.sign(emg_envelope - mav / 2)))) / (2 * len(emg_envelope))

    # Calculate the difference absolute mean value (DAMV) feature
    damv = np.median(np.abs(emg_envelope - mav))

    # Calculate frequency domain features using Fourier transform
    # fft = np.fft.rfft(emg_envelope)
    # power_spectrum = np.abs(fft) ** 2
    # total_power = np.sum(power_spectrum)
    # mean_frequency = np.sum(power_spectrum * np.arange(len(fft))) / total_power
    # spectral_centroid = sp.signal.spectral_centroid(emg_envelope, sfreq)[0]
    
    # Add any additional feature extraction methods here
    
    return[mav, rms, var, wl, zc, mav_slope, ncm, damv]


features_list = ['emg_envelope', 'mav', 'rms', 'var', 'wl', 'zc', 'mav_slope', 'ncm', 'damv']

for feature in features_list:
    # Extract features using the selected feature extraction method
    X = np.array([features(emg) for emg in emg_envelope])

    # Train a classifier
    tar = ["Switch", "Freeze", "On/Off", "Forw", "Back", "Left", "Right", "Up", "Down"]
    n = 10
    target = np.repeat(tar, len(li)//len(tar))
    y = target
    clf = SVC(kernel="linear", C=0.025)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    clf.fit(X_train, y_train)

    # Evaluate classification performance
    score = clf.score(X_test, y_test)
    print('Classification score for', feature, ':', score*100, '%')



# tar = ["Switch", "Freeze", "On/Off", "Forw", "Back", "Left", "Right", "Up", "Down"]
# n = 10
# target = np.repeat(tar, n)
# signal = emg_envelope

# X = signal
# y = target
# clf = SVC(kernel="linear", C=0.025)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
# clf.fit(X_train, y_train)
# score = clf.score(X_test, y_test)
# print(score*100, '%')
# unique_targets = np.unique(y_test)

# y_pred = clf.predict(X_test)
# unique_predictions = np.unique(y_pred)

# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(5.5, 5.5))
# percentages = cm / cm.sum(axis=1)[:, np.newaxis] * 100  # Fix here
# sns.heatmap(percentages, annot=True, cmap="Greens", fmt='.1f', xticklabels=str(tar), yticklabels=str(tar),vmin=0, vmax=100, cbar=False)
# plt.title('Iva / SVM / mav')
# plt.ylabel('Expected')
# plt.xlabel('Predicted')
# ax = plt.gca()
# ax.set_xticklabels(tar, rotation=90, ha='center')
# ax.set_yticklabels(tar, rotation=0, va='center')
# plt.show()