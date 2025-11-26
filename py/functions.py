#imports
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#all helper functions used below
def data_grabber():
    df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, "bryancimo/sdss-galaxy-classification-dr18", "sdss_100k_galaxy_form_burst.csv")
    return df
