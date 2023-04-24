from collections import deque
from threading import Lock, Thread
import joblib
import scipy as sp
import socket
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

  def update_plot(self):
    emg_data = self.listener.get_emg_data()
    while len(emg_data) < self.n:
        emg_data = self.listener.get_emg_data()
    if len(emg_data) == self.n:
      print('Checking gesture')
      emg_data = np.array([x[1] for x in emg_data]).T
      
      
      #---------------------------------------------------------------------------------------------------------------
      classifier = joblib.load('EMG_Classifier.sav')
      emg_iva = emg_data                                                            #
      for d in emg_iva:                                                             #
        flat_list = []                                                              #
        for item in d:        
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

        if len(emg_envelope) >= 1365:
            classi = classifier.predict(emg_envelope.reshape(1,-1))
            if classi == ['Forw']:
                print('Forwards')
                # HOST = "192.168.8.50"  # IP address of turtlebot robot
                # PORT = 3020  # port number for socket communication
                # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # sock.connect((HOST, PORT))
                # message = "forward".encode()
                # sock.send(message)
                # sock.close()
                sleep(1)

            if classi == ['Left']:
                print('Left')
                # HOST = "192.168.8.50"  # IP address of turtlebot robot
                # PORT = 3020  # port number for socket communication
                # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # sock.connect((HOST, PORT))
                # message = "left".encode()
                # sock.send(message)
                # sock.close()
                sleep(1)

            if classi == ['Right']:
                print('Right')
                # HOST = "192.168.8.50"  # IP address of turtlebot robot
                # PORT = 3020  # port number for socket communication
                # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # sock.connect((HOST, PORT))
                # message = "right".encode()
                # sock.send(message)
                # sock.close()
                sleep(1)

            if classi == ['Switch']:
                print('Switch')
                # HOST = "192.168.8.50"  # IP address of turtlebot robot
                # PORT = 3020  # port number for socket communication
                # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # sock.connect((HOST, PORT))
                # message = "switch".encode()
                # sock.send(message)
                # sock.close()
                sleep(1)

            if classi == ['Back']:
                print('Backwards')
                # HOST = "192.168.8.50"  # IP address of turtlebot robot
                # PORT = 3020  # port number for socket communication
                # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # sock.connect((HOST, PORT))
                # message = "station".encode()
                # sock.send(message)
                # sock.close()
                sleep(1)

            if classi == ['Up']:
                print('Up')
                # HOST = "192.168.8.50"  # IP address of turtlebot robot
                # PORT = 3020  # port number for socket communication
                # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # sock.connect((HOST, PORT))
                # message = "station".encode()
                # sock.send(message)
                # sock.close()
                sleep(1)
            
            if classi == ['Down']:
                print('Down')
                # HOST = "192.168.8.50"  # IP address of turtlebot robot
                # PORT = 3020  # port number for socket communication
                # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # sock.connect((HOST, PORT))
                # message = "station".encode()
                # sock.send(message)
                # sock.close()
                sleep(1)
            
            if classi == ['Freeze']:
                print('Freeze')
                # HOST = "192.168.8.50"  # IP address of turtlebot robot
                # PORT = 3020  # port number for socket communication
                # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # sock.connect((HOST, PORT))
                # message = "station".encode()
                # sock.send(message)
                # sock.close()
                sleep(1)



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
