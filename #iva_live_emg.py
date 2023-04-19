from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread
import joblib
import scipy as sp
from scipy.signal import filtfilt

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
    self.fig = plt.figure()
    #self.axes = [self.fig.add_subplot('81' + str(i)) for i in range(1, 9)]
    self.axes = [self.fig.add_subplot(8, 1, i) for i in range(1, 9)]              # Fixed code---------------------
    [(ax.set_ylim([-100, 100])) for ax in self.axes]
    self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
    plt.ion()

  def update_plot(self):
    emg_data = self.listener.get_emg_data()
    while len(emg_data) < 27:
        emg_data = self.listener.get_emg_data()
    emg_data = np.array([x[1] for x in emg_data]).T
    #print(emg_data.shape)
    #---------------------------------------------------------------------------------------------------------------
    classifier = joblib.load('EMG_Classifier.sav')
    emg_iva = emg_data                                                            #
    for d in emg_iva:                                                             #
      flat_list = []                                                              #
      for item in d:        
          while len(flat_list) < 2772:                                            # 
            flat_list.append(item)                                                #
    
      emg_correctmean = flat_list - np.mean(flat_list, axis=0)
      low_pass=20 # low: low-pass cut off frequency
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
      classi = classifier.predict(emg_envelope.reshape(1,-1))
      if classi == ["Switch"]:
        print("Switch") 
      if classi == ["Freeze"]:
        print("Freeze")
      if classi == ["On/Off"]:
        print("On/Off")
      if classi == ["Forwards"]:
        print("Forwards")
      if classi == ["Backwards"]:
        print("Backards")
      if classi == ["Left"]:
        print("Left")
      if classi == ["Right"]:
        print("Right")
      if classi == ["Up"]:
        print("Up")
      if classi == ["Down"]:
        print("Down")
      else:
        print("Give me an order")

      #------------------------------------------------------------------------------------------------------------
    for g, data in zip(self.graphs, emg_data):
      if len(data) < self.n:
        # Fill the left side with zeroes.
        data = np.concatenate([np.zeros(self.n - len(data)), data])
      g.set_ydata(data)
    plt.draw()


  def main(self):
    while True:
      self.update_plot()
      plt.pause(1.0 / 30)


def main():
  myo.init()
  hub = myo.Hub()
  listener = EmgCollector(2772)
  with hub.run_in_background(listener.on_event):
    Plot(listener).main()


if __name__ == '__main__':
  main()