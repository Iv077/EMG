from collections import deque
from threading import Lock, Thread
import joblib
import scipy as sp
from scipy.signal import filtfilt
import socket
import pandas as pd     #   library to be able to access each of the csv files
from time import sleep

import myo
import numpy as np


class EmgCollector(myo.DeviceListener):
  """
  Collects EMG data in a queue with *n* maximum number of elements.
  """

  def __init__(self, n):
    self.n = n
    self.lock = Lock()
    self.emg_data_queue = deque(maxlen=n)

  def get_emg_data(self):
    with self.lock:
      return list(self.emg_data_queue)

  # myo.DeviceListener

  def on_connected(self, event):
    event.device.stream_emg(True)

  def on_emg(self, event):
    with self.lock:
      self.emg_data_queue.append((event.timestamp, event.emg))


class Plot(object):

  def __init__(self, listener):
    self.n = listener.n
    self.listener = listener
    self.start_collecting = False
    self.collected_data = []
    self.last_timestamp = 0
    self.classifier = joblib.load('EMG_Classifier.sav')

  def update_plot(self):
    emg_data = self.listener.get_emg_data()
    oemg_data = np.array([x[1] for x in emg_data]).T

    if not self.start_collecting:
      if np.any(oemg_data >= 50):
        self.start_collecting = True
        self.collected_data = []
        print('Start collecting data for classification.')
      else:
        return


    if len(emg_data) > 0:
      timestamp, emg_list = emg_data[-1]
      if timestamp != self.last_timestamp:
        self.last_timestamp = timestamp

        self.collected_data.append(emg_list)
        print(self.collected_data)

      #emg_list = np.array(emg_list)
      # Collect data for classification
      # self.collected_data.append(emg_list)
      count = 200
      for d in emg_list:                                                                                                                    #          
          self.collected_data.append(d)
          # count += 1
          # if count == 200:
          #     break
          #print(self.collected_data)


    # When the needed number of values is collected, perform classification
    if len(self.collected_data) == 2730:                                                
      emg_iva = np.array(self.collected_data)
      
      #print(emg_iva)                                                             #
      flat_list = []                                                              #
      for item in emg_iva:        
          flat_list.append(item)                                                  #
      
      emg_correctmean = flat_list - np.mean(flat_list, axis=0)
      low_pass=40 # low: low-pass cut off frequency
      sfreq=1000 # sfreq: sampling frequency
      high_band=40
      low_band=450
      # emg: EMG data
      # high: high-pass cut off frequency
      
      # normalise cut-off frequencies to sampling frequency
      high_band = high_band/(sfreq/2)
      low_band = low_band/(sfreq/2)

      # create bandpass filter for EMG
      b1, a1 = sp.signal.butter(4, [high_band,low_band], btype='bandpass')

      # process EMG signal: filter EMG
      emg_filtered = sp.signal.filtfilt(b1, a1, emg_correctmean, axis=0)

      # process EMG signal: rectify
      emg_rectified = abs(emg_filtered)

      # create lowpass filter and apply to rectified signal to get EMG envelope
      low_pass = low_pass/(sfreq/2)
      b2, a2 = sp.signal.butter(4, low_pass, btype='lowpass')
      emg_envelope = np.array(sp.signal.filtfilt(b2, a2, emg_rectified, axis=0))

      classi = self.classifier.predict(emg_envelope.reshape(1,-1))

    self.start_collecting = False
    self.collected_data = []
    #print('Classification result:', classi)


      #------------------------------------------------------------------------------------------------------------


  def main(self):
    while True:
      self.update_plot()

def main():
  myo.init()
  hub = myo.Hub()
  listener = EmgCollector(2730)
  with hub.run_in_background(listener.on_event):
    Plot(listener).main()


if __name__ == '__main__':
  main()
