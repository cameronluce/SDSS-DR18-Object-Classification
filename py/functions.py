#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter


#all helper functions used
url = "https://www.kaggle.com/api/v1/datasets/download/bryancimo/sdss-galaxy-classification-dr18"

def data_grabber():
    data = pd.read_csv(url)
    return data
