from time import time
from collections import deque
from threading import Lock
import numpy as np
import pandas as pd

import myo

import os

from tkinter import *
from tkinter import ttk
import joblib

class Gui(object):
    def __init__(self):
        self.record = True
        self.record_label = 'default_label'
        self.record_file_name = 'default_file'
        self.timestamp_data = deque(maxlen=1)
        self.emg_data = deque(maxlen=1)
        self.record_text = ''
        pass

        print(self.emg_data[0][i])