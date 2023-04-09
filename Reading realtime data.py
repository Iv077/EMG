from time import time
from collections import deque
from threading import Lock

import myo

import os

from tkinter import *
from tkinter import ttk

class Gui(object):
  def __init__(self):
    self.record = False
    self.record_label = 'default_label'
    self.record_file_name = 'default_file'
    self.start_time = 0
    self.record_time_start = 0
    self.record_time = 20
    self.timestamp_data = deque(maxlen=1)
    self.emg_data = deque(maxlen=1)
    self.record_text = ''
    pass
  
  def gui_config(self):
      self.title = self.mainWin.title("Myo Armband Emg Data Recorder")
      self.mainWin.geometry("400x200")
      self.mainWin.resizable(width = FALSE, height= FALSE)
      self.appWin = ttk.Frame(self.mainWin)
      pass
  
  def gui_widgets(self):


      # Myo Record Button
      self.button_myo_record_data = ttk.Button(text="Start Record", master=self.appWin, command=self.start_record_data) #show_emg_data #start_reading_emg_sensors
      self.button_myo_record_data.grid(row=3,column=0)

      # Myo Stop Record Button
      self.button_myo_data_record_stop = ttk.Button(text="Stop Record", master=self.appWin, command=self.stop_record_data, state=DISABLED)
      self.button_myo_data_record_stop.grid(row=4,column=0)

      # Record Status Label
      self.label_myo_record_status = ttk.Label(text="", master=self.appWin, foreground='black')
      self.label_myo_record_status.grid(row=6, columnspan=1)
    
      self.appWin.pack()

      pass

  def record_data(self):
    if self.record:
      dif_time = time() - self.record_time_start
      if  dif_time < self.record_time:
        record_time = str(int(dif_time*1000))
        self.label_myo_record_status.config(text= str(int(dif_time*1000)) + "ms - Recording...", foreground= 'green')
        sensor_data = ''
        for i in range (8):
          sensor_data = sensor_data + ',' + str(self.emg_data[0][i])

        label = ',' + self.record_label + '\n'
        try:
          self.record_file.write(record_time + sensor_data + label)
        except Exception:
          pass
        print(sensor_data)

      else:
        self.label_myo_record_status.config(text= "Recording done.", foreground='black')
        self.button_myo_record_data.state(["!disabled"])
        self.button_myo_data_record_stop.state(['disabled'])
        self.record = False
        self.record_file.close()
  
  def start_record_data(self):
    self.record_file = open( os.path.dirname(__file__) + '\\' + str(self.record_file_name) + '.csv','w')
    record_headers = 'timestamp,sensor1,sensor2,sensor3,sensor4,sensor5,sensor6,sensor7,sensor8'
    
    record_headers = record_headers + ',label\n'

    self.record_file.write(record_headers)
    self.record = True
    self.button_myo_record_data.state(["disabled"])
    self.button_myo_data_record_stop.state(["!disabled"])
    self.record_time_start = time()
    pass

  def stop_record_data(self):
    self.button_myo_record_data.state(["!disabled"])
    self.button_myo_data_record_stop.state(['disabled'])
    self.label_myo_record_status.config(text= "Recording done.", foreground='black')
    self.record = False
    self.record_file.close()
    pass

  def main(self):
    self.mainWin = Tk()
    self.gui_config()
    self.gui_widgets()

class EmgCollector(myo.DeviceListener):
  
  def __init__(self, n):
    self.n = n
    self.lock = Lock()
    self.start_time = 0
    self.app = Gui()

  def get_emg_data(self):
    with self.lock:
      return list(self.emg_data_queue)

  def on_connected(self, event):
    print("Myo connected...")
    self.app.label_myo_record_status.config(text="Myo Armband Status: Connected",foreground= 'green')
    self.myo_status = True
    event.device.stream_emg(True)

  def on_disconnected(self, event):
    print("Myo disconnected...")
    self.app.label_myo_record_status.config(text="Myo Armband Status: Disconnected",foreground= 'red')
    self.myo_status = False

  def on_emg(self, event):
    with self.lock:
      if self.start_time == 0:
        self.start_time = event.timestamp
        self.app.start_time = self.start_time
      
      self.app.timestamp_data.append(int((event.timestamp - self.start_time)/1000))
      self.app.emg_data.append(event.emg)

      self.app.record_text = str(self.app.timestamp_data[0]) + " " + str(event.emg)

      self.app.record_text = self.app.record_text + " " + self.app.record_label

      self.app.record_data()
  
  
  def main(self):
    self.app.main()
    myo.init(sdk_path= os.path.dirname(__file__) + '\\myo-sdk-win-0.9.0')
    hub = myo.Hub()
    with hub.run_in_background(self.on_event):
      mainloop()
      
def main():
    listener = EmgCollector(1)
    listener.main()


if __name__ == '__main__':
    main()