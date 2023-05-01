from collections import deque
from threading import Lock, Thread
import joblib
import scipy as sp
from scipy.signal import filtfilt
import socket
import pandas as pd     #   library to be able to access each of the csv files
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import myo
import numpy as np


class EmgCollector(myo.DeviceListener):
  """
  Collects EMG data in a queue with *n* maximum number of elements.
  """

  def __init__(self, n):
    self.n = n
    self.lock = Lock()
    self.emg_data_queue = deque(maxlen=2000)

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
    self.classifier = joblib.load('EMG_Classifier3.sav')
    self.last_timestamp = 0

  def update_plot(self):
    emg_sig = self.listener.get_emg_data()
    emg_sig = np.array([x[1] for x in emg_sig]).T
    if not self.start_collecting:
      print('Expecting a gesture')
      if np.any(emg_sig >= 50):
        self.start_collecting = True
        self.collected_data = []
        #print('Start collecting data for classification.')
      else:
        return
      
      
    # Collect data for classification
    if self.start_collecting:   # only collect data if start_collecting is True
      #print('Recording the gesture')
      emg_data = self.listener.get_emg_data()
      #emg_data = np.array([x[1] for x in emg_data]).T
      if len(emg_data) > 0:
        timestamp, emg_list = emg_data[-1]
        if timestamp != self.last_timestamp:
          self.last_timestamp = timestamp
        
          self.collected_data.append(emg_list)
          if len(self.collected_data)== 342:
            hi = []
            for d in self.collected_data:
                for i in d:
                    hi.append(i)
            #print(len(hi))


    # When the needed number of values is collected, perform classification
            if len(hi) >= 2730:                                                
                emg_iva = hi [:2730]
                print('I got the gesture, relax')
                #print(emg_iva)
                
                emg_correctmean = emg_iva - np.mean(emg_iva, axis=0)
                low_pass=10 # low: low-pass cut off frequency
                sfreq=400 # sfreq: sampling frequency
                high_band=20
                low_band=50
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
                #print(emg_envelope.shape)
                mav_slope = np.abs(np.diff(emg_envelope))

                
                classi = self.classifier.predict(mav_slope.reshape(1,-1))

                #print('Classification result:', classi)
                if classi == ['Horn']:
                  print('Forwards/Horn')
                  # HOST = "192.168.8.50"  # IP address of turtlebot robot
                  # PORT = 2000  # port number for socket communication
                  # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                  # sock.connect((HOST, PORT))
                  # message = "forward".encode()
                  # start_time = time.perf_counter()  # start time measurement
                  # sock.send(message)
                  # sock.close()
                  # end_time = time.perf_counter()  # end time measurement
                  # time_taken = end_time - start_time
                  # print("Time taken: {:.10f} seconds".format(time_taken))
                            


                if classi == ['Victory']:
                    print('Left/Victory')
                    # HOST = "192.168.8.50"  # IP address of turtlebot robot
                    # PORT = 2000  # port number for socket communication
                    # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    # sock.connect((HOST, PORT))
                    # message = "left".encode()
                    # start_time = time.perf_counter()  # start time measurement
                    # sock.send(message)
                    # sock.close()
                    # end_time = time.perf_counter()  # end time measurement
                    # time_taken = end_time - start_time
                    # print("Time taken: {:.10f} seconds".format(time_taken))
                            

                if classi == ['Rotation']:
                    print('Right/Rotation')
                    # HOST = "192.168.8.50"  # IP address of turtlebot robot
                    # PORT = 2000  # port number for socket communication
                    # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    # sock.connect((HOST, PORT))
                    # message = "right".encode()
                    # start_time = time.perf_counter()  # start time measurement
                    # sock.send(message)
                    # sock.close()
                    # end_time = time.perf_counter()  # end time measurement
                    # time_taken = end_time - start_time
                    # print("Time taken: {:.10f} seconds".format(time_taken))
                                    

                # if classi == ['Switch']:
                #     print('')
                #     HOST = "192.168.8.50"  # IP address of turtlebot robot
                #     PORT = 3020  # port number for socket communication
                #     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                #     sock.connect((HOST, PORT))
                #     message = "right".encode()
                #     sock.send(message)
                #     sock.close()
                    

                # if classi == ['Back']:
                #     print('Backwards')
                #     # HOST = "192.168.8.50"  # IP address of turtlebot robot
                #     # PORT = 3020  # port number for socket communication
                #     # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                #     # sock.connect((HOST, PORT))
                #     # message = "station".encode()
                #     # sock.send(message)
                #     # sock.close()


                # if classi == ['Up']:
                #     print('Up')
                #     # HOST = "192.168.8.50"  # IP address of turtlebot robot
                #     # PORT = 3020  # port number for socket communication
                #     # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                #     # sock.connect((HOST, PORT))
                #     # message = "station".encode()
                #     # sock.send(message)
                #     # sock.close()
                
                # if classi == ['Down']:
                #     print('Down')
                #     # HOST = "192.168.8.50"  # IP address of turtlebot robot
                #     # PORT = 3020  # port number for socket communication
                #     # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                #     # sock.connect((HOST, PORT))
                #     # message = "station".encode()
                #     # sock.send(message)
                #     # sock.close()
                
                # if classi == ['Freeze']:
                #     print('Freeze')
                #     # HOST = "192.168.8.50"  # IP address of turtlebot robot
                #     # PORT = 3020  # port number for socket communication
                #     # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                #     # sock.connect((HOST, PORT))
                #     # message = "station".encode()
                #     # sock.send(message)
                #     # sock.close()


                self.start_collecting = False
                return
                


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
